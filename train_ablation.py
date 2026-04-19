from __future__ import annotations

import os
import json
import time
import argparse
from collections import deque
from typing import Callable

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


MAX_STEPS = 1000
EVAL_EPISODES = 5


class StepLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps: int = 1000):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, done, truncated, info


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k: int = 3):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        img_space = env.observation_space["image"]
        shp = img_space.shape
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs["image"])
        return self._get_obs(obs), info

    def observation(self, obs):
        self.frames.append(obs["image"])
        return self._get_obs(obs)

    def _get_obs(self, obs):
        return {
            "image": np.concatenate(list(self.frames), axis=0),
            "features": obs["features"],
        }


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        self.shape = shape

        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, shape[0], shape[1]),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, obs):
        img = obs["image"]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        resized = cv2.resize(img, self.shape)
        return {
            "image": resized.transpose(2, 0, 1),
            "features": obs["features"],
        }


class DictObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space
        feat_dim = env.unwrapped.observation_space.shape[0]
        self.observation_space = gym.spaces.Dict(
            {
                "image": img_space,
                "features": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(feat_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, obs):
        return {"image": obs, "features": self._last_features}

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_features = info.get(
            "proprio_obs", self.env.unwrapped._get_obs()
        ).astype(np.float32)
        return self.observation(raw_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_features = self.env.unwrapped._get_obs().astype(np.float32)
        return self.observation(obs), info


class CurriculumRewardWrapper(gym.Wrapper):
    """
    Reward shaping usado SOLO para entrenamiento cuando se activa.
    No debe usarse en evaluaci�n comparativa si queremos rewards comparables.
    """
    def __init__(self, env, total_steps: int = 100_000):
        super().__init__(env)
        self.total_steps = total_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.current_step += 1
        alpha = min(1.0, (self.current_step / self.total_steps) ** 2)

        features = obs["features"]
        tips_arm = features[14:17]
        obj_pos = features[17:20]
        goal_pos = features[20:23]

        dist_hand_obj = np.linalg.norm(tips_arm - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)

        reach_reward = -dist_hand_obj * 5.0
        push_reward = -dist_obj_goal * 5.0
        reward = (1.0 - alpha) * reach_reward + alpha * push_reward

        if dist_hand_obj < 0.1:
            reward += 5.0
        if dist_obj_goal < 0.1:
            reward += 20.0

        reward -= 0.001 * np.sum(np.square(action))
        return obs, reward, done, truncated, info


class FeaturesOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["features"]

    def observation(self, obs):
        return obs["features"]


class ImagesOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        mlp_output_dim: int = 64,
    ):
        super().__init__(
            observation_space,
            features_dim=cnn_output_dim + mlp_output_dim,
        )

        img_space = observation_space["image"]
        feat_space = observation_space["features"]
        n_ch = img_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(img_space.sample()[None]).float()
            n_flat = self.cnn(sample).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flat, cnn_output_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(feat_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        img = observations["image"].float() / 255.0
        feat = observations["features"].float()
        return torch.cat([self.cnn_linear(self.cnn(img)), self.mlp(feat)], dim=1)


class CNNOnlyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=cnn_output_dim)
        n_ch = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, cnn_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        img = observations.float() / 255.0
        return self.linear(self.cnn(img))


class MetricsCallback(BaseCallback):
    """
    Guarda episodios para curva PNG y deja a SB3 escribir el resto en TensorBoard.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.timesteps_at_ep = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.timesteps_at_ep.append(self.num_timesteps)
        return True


def make_env(
    variant: str,
    total_steps: int,
    img_size: int = 64,
    use_curriculum: bool = True,
) -> Callable[[], gym.Env]:
    """
    variant controla SOLO la observaci�n / pol�tica.
    use_curriculum controla SOLO la recompensa del entorno.

    Para entrenamiento:
      - multimodal_curriculum       -> use_curriculum=True
      - multimodal_no_curriculum    -> use_curriculum=False
      - features_only               -> normalmente True
      - images_only                 -> normalmente True

    Para evaluaci�n COMPARABLE:
      - usar use_curriculum=False en todos.
    """
    def _init():
        env = gym.make("Pusher-v5", render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env)
        env = DictObservationWrapper(env)
        env = ResizeObservation(env, (img_size, img_size))
        env = FrameStack(env, k=3)

        if use_curriculum:
            env = CurriculumRewardWrapper(env, total_steps=total_steps)

        if variant == "features_only":
            env = FeaturesOnlyWrapper(env)
        elif variant == "images_only":
            env = ImagesOnlyWrapper(env)

        env = StepLimitWrapper(env, max_steps=MAX_STEPS)
        env = Monitor(env)
        return env

    return _init


def save_learning_curve(cb: MetricsCallback, out_path: str, title: str):
    if not cb.episode_rewards:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        cb.timesteps_at_ep,
        cb.episode_rewards,
        alpha=0.35,
        linewidth=0.8,
        color="#90CAF9",
    )

    if len(cb.episode_rewards) >= 10:
        window = max(10, len(cb.episode_rewards) // 20)
        smoothed = np.convolve(
            cb.episode_rewards, np.ones(window) / window, mode="valid"
        )
        x_sm = cb.timesteps_at_ep[window - 1 :]
        ax.plot(
            x_sm,
            smoothed,
            linewidth=2,
            color="#1565C0",
            label="Smoothed",
        )
        ax.legend()

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def build_model(variant: str, env, device: str, tensorboard_root: str):
    common_kwargs = dict(
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=100_000,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_root,
        device=device,
    )

    if variant in ("multimodal_curriculum", "multimodal_no_curriculum"):
        return SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=CombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=256, mlp_output_dim=64),
            ),
            **common_kwargs,
        )

    if variant == "images_only":
        return SAC(
            "CnnPolicy",
            env,
            policy_kwargs=dict(
                features_extractor_class=CNNOnlyExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=256),
            ),
            **common_kwargs,
        )

    if variant == "features_only":
        return SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            **common_kwargs,
        )

    raise ValueError(f"Unknown variant: {variant}")


def train_variant(
    variant: str,
    out_dir: str,
    timesteps: int,
    img_size: int,
    device: str,
    eval_common_reward: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # Entrenamiento: solo multimodal_no_curriculum entrena sin shaping.
    train_use_curriculum = variant != "multimodal_no_curriculum"

    train_env = DummyVecEnv(
        [make_env(variant, timesteps, img_size=img_size, use_curriculum=train_use_curriculum)]
    )
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    # Evaluaci�n comparable:
    # - mismo espacio de observaci�n que el modelo
    # - MISMA recompensa para todos si eval_common_reward=True
    #   (sin curriculum reward)
    eval_use_curriculum = not eval_common_reward
    eval_env = DummyVecEnv(
        [make_env(variant, timesteps, img_size=img_size, use_curriculum=eval_use_curriculum)]
    )
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    tb_root = os.path.join(out_dir, "tb")
    os.makedirs(tb_root, exist_ok=True)

    model = build_model(
        variant=variant,
        env=train_env,
        device=device,
        tensorboard_root=tb_root,
    )

    metrics_cb = MetricsCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=max(timesteps // 10, 5000),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[metrics_cb, eval_cb],
        tb_log_name=variant,
    )
    elapsed = time.time() - t0

    mean_r, std_r = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        return_episode_rewards=False,
    )

    model.save(os.path.join(out_dir, "model_final"))
    train_env.save(os.path.join(out_dir, "vecnorm.pkl"))
    save_learning_curve(
        metrics_cb,
        os.path.join(out_dir, "learning_curve.png"),
        f"Ablation - {variant}",
    )

    train_env.close()
    eval_env.close()

    return {
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "elapsed_sec": round(elapsed, 1),
        "n_episodes": len(metrics_cb.episode_rewards),
        "final_ep_rew": float(np.mean(metrics_cb.episode_rewards[-20:]))
        if metrics_cb.episode_rewards
        else 0.0,
        "train_use_curriculum": bool(train_use_curriculum),
        "eval_use_curriculum": bool(eval_use_curriculum),
        "eval_common_reward": bool(eval_common_reward),
    }


def save_ablation_plot(results, out_path: str):
    labels = [r["variant"] for r in results]
    means = [r["metrics"]["mean_reward"] for r in results]
    stds = [r["metrics"]["std_reward"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"][: len(labels)]

    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean eval reward")
    ax.set_title("Ablation study - final evaluation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for SAC Pusher")
    parser.add_argument("--out_dir", type=str, default="./ablation_results")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval_common_reward",
        action="store_true",
        help="Evaluate all variants with the same reward definition (no curriculum reward). Recommended for fair comparison.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "multimodal_curriculum",
            "multimodal_no_curriculum",
            "features_only",
            "images_only",
        ],
        choices=[
            "multimodal_curriculum",
            "multimodal_no_curriculum",
            "features_only",
            "images_only",
        ],
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        run_dir = os.path.join(args.out_dir, variant)
        metrics = train_variant(
            variant=variant,
            out_dir=run_dir,
            timesteps=args.timesteps,
            img_size=args.img_size,
            device=device,
            eval_common_reward=args.eval_common_reward,
        )
        all_results.append({"variant": variant, "metrics": metrics})
        print(
            f"mean_reward={metrics['mean_reward']:.2f} � {metrics['std_reward']:.2f} | "
            f"time={metrics['elapsed_sec']:.0f}s"
        )

    summary_path = os.path.join(args.out_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    save_ablation_plot(all_results, os.path.join(args.out_dir, "ablation_comparison.png"))
    print(f"[saved] {summary_path}")
    print(
        "\nTensorBoard example:\n"
        f"  tensorboard --logdir {args.out_dir} --port 6006\n"
    )


if __name__ == "__main__":
    main()