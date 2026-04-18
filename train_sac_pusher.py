import os
import gymnasium as gym
import numpy as np
import cv2
import imageio
from collections import deque
 
import matplotlib.pyplot as plt
 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
 
import torch
import torch.nn as nn
 
 
MAX_STEPS = 1000
 
 
# -------- Step Limit Wrapper --------
class StepLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=1000):
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
 
 
# -------- Frame Stack Wrapper (solo para imagen dentro del Dict) --------
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
 
        img_space = env.observation_space["image"]
        shp = img_space.shape  # (C, H, W)
 
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=np.uint8
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
            "features": obs["features"]
        }
 
 
# -------- Resize Wrapper (solo para imagen dentro del Dict) --------
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(128, 128)):
        super().__init__(env)
        self.shape = shape
 
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255,
            shape=(3, shape[0], shape[1]),
            dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)
 
    def observation(self, obs):
        img = obs["image"]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        resized = cv2.resize(img, self.shape)
        return {
            "image": resized.transpose(2, 0, 1),
            "features": obs["features"]
        }
 
 
# -------- Dict Observation Wrapper --------
# Separa la observación nativa en imagen (render) + features (estado propioceptivo)
class DictObservationWrapper(gym.ObservationWrapper):
    """
    Transforma el espacio de observación en un Dict con:
      - "image": observación visual (render adjuntado por AddRenderObservation)
      - "features": observación propioceptiva original del entorno
    """
    def __init__(self, env):
        super().__init__(env)
 
        # AddRenderObservation añade la imagen al FINAL del espacio Box.
        # Para Pusher-v5 la obs propioceptiva tiene shape (23,), y el render
        # se adjunta como canal visual separado. Sin embargo, con render_mode="rgb_array"
        # y AddRenderObservation el espacio resultante es un Box con imagen.
        # Aquí asumimos que env.observation_space ya es el espacio de imagen
        # porque AddRenderObservation se aplica ANTES.
        img_space = env.observation_space  # (3, H, W) tras AddRenderObservation
 
        # Obtenemos el espacio propioceptivo del entorno base (sin wrappers de render)
        raw_env = env.unwrapped
        feat_dim = raw_env.observation_space.shape[0]  # 23 para Pusher-v5
 
        self.observation_space = gym.spaces.Dict({
            "image": img_space,
            "features": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(feat_dim,),
                dtype=np.float32
            )
        })
 
    def observation(self, obs):
        # obs aquí ya es la imagen (C,H,W) gracias a AddRenderObservation
        # Recuperamos las features propioceptivas del info que guarda el wrapper
        return {
            "image": obs,
            "features": self._last_features
        }
 
    def step(self, action):
        # Interceptamos step para capturar las features propioceptivas ANTES
        # de que AddRenderObservation las descarte
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        # Las features propioceptivas las guardamos en info por el env base
        self._last_features = info.get(
            "proprio_obs",
            self.env.unwrapped._get_obs()  # fallback directo al env
        ).astype(np.float32)
        obs = self.observation(raw_obs)
        return obs, reward, terminated, truncated, info
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_features = self.env.unwrapped._get_obs().astype(np.float32)
        return self.observation(obs), info
 
 
# -------- Curriculum Reward Wrapper (compatible con Dict obs) --------
class CurriculumRewardWrapper(gym.Wrapper):
    def __init__(self, env, total_steps=300000):
        super().__init__(env)
        self.total_steps = total_steps
        self.current_step = 0
 
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
 
        self.current_step += 1
        alpha = min(1.0, (self.current_step / self.total_steps) ** 2)
 
        # Accedemos a las features propioceptivas desde el dict de observación
        features = obs["features"]
 
        tips_arm = features[14:17]
        obj_pos  = features[17:20]
        goal_pos = features[20:23]
 
        dist_hand_obj = np.linalg.norm(tips_arm - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
 
        reach_reward = -dist_hand_obj * 5
        push_reward  = -dist_obj_goal * 5
 
        reward = (1 - alpha) * reach_reward + alpha * push_reward
 
        if dist_hand_obj < 0.1:
            reward += 5
 
        if dist_obj_goal < 0.1:
            reward += 20
 
        reward -= 0.001 * np.sum(np.square(action))
 
        return obs, reward, done, truncated, info
 
 
# -------- Custom Combined Feature Extractor (CNN + MLP) --------
class CombinedExtractor(BaseFeaturesExtractor):
    """
    Extractor multimodal:
      - CNN para la observación visual (imagen)
      - MLP para las features propioceptivas
    Las salidas se concatenan para producir el vector de features final.
    """
    def __init__(self, observation_space: gym.spaces.Dict,
                 cnn_output_dim: int = 256,
                 mlp_output_dim: int = 64):
 
        # features_dim = cnn_output_dim + mlp_output_dim
        super().__init__(observation_space, features_dim=cnn_output_dim + mlp_output_dim)
 
        img_space = observation_space["image"]
        feat_space = observation_space["features"]
 
        n_input_channels = img_space.shape[0]
 
        # --- CNN para imagen ---
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
 
        with torch.no_grad():
            sample = torch.as_tensor(img_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
 
        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )
 
        # --- MLP para features propioceptivas ---
        feat_dim = feat_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim),
            nn.ReLU()
        )
 
    def forward(self, observations):
        # Imagen
        img = observations["image"].float() / 255.0
        cnn_out = self.cnn_linear(self.cnn(img))
 
        # Features propioceptivas
        feat = observations["features"].float()
        mlp_out = self.mlp(feat)
 
        # Concatenar
        return torch.cat([cnn_out, mlp_out], dim=1)
 
 
# -------- Video Recorder Callback (GIF) --------
class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_folder="./videos/pusher/sac",
                 record_every=10000, verbose=0):
        super().__init__(verbose)
        self.video_folder = video_folder
        self.record_every = record_every
        self._last_recorded = 0
        os.makedirs(video_folder, exist_ok=True)
 
    def _make_eval_env(self):
        env = gym.make("Pusher-v5", render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env)
        env = DictObservationWrapper(env)
        env = ResizeObservation(env, (128, 128))
        env = FrameStack(env, k=3)
        return env
 
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_recorded >= self.record_every:
            self._last_recorded = self.num_timesteps
            self._record_video()
        return True
 
    def _record_video(self):
        step_str = f"{self.num_timesteps:07d}"
 
        env = self._make_eval_env()
        obs, _ = env.reset()
        raw_env = env.unwrapped
 
        frames = []
        done = False
        step = 0
 
        while not done and step < MAX_STEPS:
            frame = raw_env.render()
 
            if step % 2 == 0:
                frames.append(frame)
 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
 
        env.close()
 
        if not frames:
            return
 
        TARGET_H = 240
        h0, w0 = frames[0].shape[:2]
        scale = TARGET_H / h0
        new_w = int(w0 * scale)
 
        resized_frames = [
            cv2.resize(frame, (new_w, TARGET_H))
            for frame in frames
        ]
 
        gif_path = os.path.join(self.video_folder, f"step_{step_str}.gif")
 
        imageio.mimsave(gif_path, resized_frames, fps=15)
 
        if self.verbose:
            print(f"[VideoRecorder] Saved GIF: {gif_path} ({len(frames)} frames)")
 
 
# -------- TensorBoard Callback --------
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
 
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.logger.record("custom/episode_reward", info["episode"]["r"])
                self.logger.record("custom/episode_length", info["episode"]["l"])
        return True
 
 
# -------- Train Env --------
def make_train_env(total_steps):
    def _init():
        env = gym.make("Pusher-v5", render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env)
        env = DictObservationWrapper(env)      # -> Dict{image, features}
        env = ResizeObservation(env, (128, 128))
        env = FrameStack(env, k=3)
        env = CurriculumRewardWrapper(env, total_steps=total_steps)
        env = StepLimitWrapper(env, max_steps=MAX_STEPS)
        env = Monitor(env)
        return env
 
    return _init
 
 
# -------- MAIN --------
def main():
    TOTAL_TIMESTEPS = 300_000
 
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(
            cnn_output_dim=256,
            mlp_output_dim=64
        )
    )
 
    train_env = DummyVecEnv([make_train_env(TOTAL_TIMESTEPS)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
 
    model = SAC(
        "MultiInputPolicy",       
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=5000,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log="./runs/",
        device="cuda"
    )
 
    video_cb = VideoRecorderCallback(
        video_folder="./videos/pusher/sac",
        record_every=10000,
        verbose=1
    )
 
    tensorboard_cb = TensorboardCallback()
 
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[video_cb, tensorboard_cb]
    )
 
    model.save("sac_pusher_combined")
    train_env.save("sac_pusher_vecnormalize.pkl")
 
    print("Training complete. GIFs saved in ./videos/pusher/sac")
 
 
if __name__ == "__main__":
    main()