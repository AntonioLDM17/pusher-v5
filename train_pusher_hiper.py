"""
hyperparam_grid.py
==================
Grid search manual de hiperparámetros para SAC Pusher (imagen + features).
 
Define los valores a explorar en PARAM_GRID y el script entrena una run
por cada combinación, guardando métricas, checkpoints y un resumen final.
 
Uso:
    python hyperparam_grid.py                          # usa el grid por defecto
    python hyperparam_grid.py --parallel 2             # 2 runs en paralelo (subprocesos)
    python hyperparam_grid.py --dry_run                # imprime combinaciones sin entrenar
    python hyperparam_grid.py --resume ./grid_results  # salta combinaciones ya completadas
"""
 
import os
import json
import time
import copy
import argparse
import itertools
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
 
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
 
 
# ═══════════════════════════════════════════════════════════════
#  ▶  EDITA AQUÍ TU GRID  ◀
#     Cada lista define los valores a probar para ese parámetro.
#     El script entrena todas las combinaciones posibles.
# ═══════════════════════════════════════════════════════════════
 
PARAM_GRID = {
    # SAC core
    "learning_rate":    [1e-3, 1e-4],
    "batch_size":       [256],
    "buffer_size":      [100_000],          # un solo valor = fijo para todos
    "learning_starts":  [5_000],
    "train_freq":       [1],
    "gradient_steps":   [1],
 
    # Arquitectura del extractor
    "cnn_output_dim":   [256],
    "mlp_output_dim":   [32, 64],
 
    # Curriculum
    "curriculum_steps": [300_000],
 
    # Entorno
    "frame_stack_k":    [3],
    "img_size":         [64, 128],              # resolución cuadrada (N → N×N)
}
 
# Timesteps de entrenamiento por combinación (reducir para búsqueda rápida)
TIMESTEPS_PER_RUN = 300_000
 
# Episodios de evaluación final (después de entrenar)
EVAL_EPISODES = 5
 
MAX_STEPS = 1000
 
 
# ═══════════════════════════════════════════════════════════════
#  Wrappers (idénticos al script de entrenamiento principal)
# ═══════════════════════════════════════════════════════════════
 
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
 
 
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k=3):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        img_space = env.observation_space["image"]
        shp = img_space.shape
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8
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
        return {"image": np.concatenate(list(self.frames), axis=0),
                "features": obs["features"]}
 
 
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(128, 128)):
        super().__init__(env)
        self.shape = shape
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255, shape=(3, shape[0], shape[1]), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)
 
    def observation(self, obs):
        img = obs["image"]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        resized = cv2.resize(img, self.shape)
        return {"image": resized.transpose(2, 0, 1), "features": obs["features"]}
 
 
class DictObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space
        feat_dim  = env.unwrapped.observation_space.shape[0]
        self.observation_space = gym.spaces.Dict({
            "image": img_space,
            "features": gym.spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(feat_dim,), dtype=np.float32)
        })
 
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
    def __init__(self, env, total_steps=300_000):
        super().__init__(env)
        self.total_steps = total_steps
        self.current_step = 0
 
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.current_step += 1
        alpha = min(1.0, (self.current_step / self.total_steps) ** 2)
 
        features      = obs["features"]
        tips_arm      = features[14:17]
        obj_pos       = features[17:20]
        goal_pos      = features[20:23]
        dist_hand_obj = np.linalg.norm(tips_arm - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
 
        reach_reward  = -dist_hand_obj * 5
        push_reward   = -dist_obj_goal * 5
        reward        = (1 - alpha) * reach_reward + alpha * push_reward
 
        if dist_hand_obj < 0.1: reward += 5
        if dist_obj_goal < 0.1: reward += 20
        reward -= 0.001 * np.sum(np.square(action))
 
        return obs, reward, done, truncated, info
 
 
# ═══════════════════════════════════════════════════════════════
#  Extractor combinado (parametrizable)
# ═══════════════════════════════════════════════════════════════
 
class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 cnn_output_dim: int = 256, mlp_output_dim: int = 64):
        super().__init__(observation_space, features_dim=cnn_output_dim + mlp_output_dim)
 
        img_space  = observation_space["image"]
        feat_space = observation_space["features"]
        n_ch       = img_space.shape[0]
 
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,    64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64,   128, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(128,  256, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.as_tensor(img_space.sample()[None]).float()).shape[1]
 
        self.cnn_linear = nn.Sequential(nn.Linear(n_flat, cnn_output_dim), nn.ReLU())
        self.mlp        = nn.Sequential(
            nn.Linear(feat_space.shape[0], 64), nn.ReLU(),
            nn.Linear(64, mlp_output_dim),      nn.ReLU()
        )
 
    def forward(self, observations):
        img  = observations["image"].float() / 255.0
        feat = observations["features"].float()
        return torch.cat([self.cnn_linear(self.cnn(img)), self.mlp(feat)], dim=1)
 
 
# ═══════════════════════════════════════════════════════════════
#  Callback: guarda métricas de episodio en una lista compartida
# ═══════════════════════════════════════════════════════════════
 
class MetricsCallback(BaseCallback):
    """Recoge episode_reward y episode_length en tiempo real."""
 
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_ep = []
 
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_ep.append(self.num_timesteps)
        return True
 
 
# ═══════════════════════════════════════════════════════════════
#  Construcción del entorno a partir de un dict de hiperparámetros
# ═══════════════════════════════════════════════════════════════
 
def make_env(hp: dict):
    def _init():
        env = gym.make("Pusher-v5", render_mode="rgb_array")
        env = gym.wrappers.AddRenderObservation(env)
        env = DictObservationWrapper(env)
        env = ResizeObservation(env, (hp["img_size"], hp["img_size"]))
        env = FrameStack(env, k=hp["frame_stack_k"])
        env = CurriculumRewardWrapper(env, total_steps=hp["curriculum_steps"])
        env = StepLimitWrapper(env, max_steps=MAX_STEPS)
        env = Monitor(env)
        return env
    return _init
 
 
# ═══════════════════════════════════════════════════════════════
#  Entrenamiento de una combinación de hiperparámetros
# ═══════════════════════════════════════════════════════════════
 
def train_one(hp: dict, run_dir: str, timesteps: int, device: str) -> dict:
    """
    Entrena un modelo con los hiperparámetros `hp`.
    Devuelve un dict con las métricas de la run.
    """
    os.makedirs(run_dir, exist_ok=True)
 
    # ── Env ───────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(hp)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
 
    eval_vec = DummyVecEnv([make_env(hp)])
    eval_env_raw = VecNormalize(eval_vec, norm_obs=False, norm_reward=False, training=False)

    
 
    # ── Modelo ───────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(
            cnn_output_dim=hp["cnn_output_dim"],
            mlp_output_dim=hp["mlp_output_dim"],
        )
    )
 
    model = SAC(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=hp["learning_rate"],
        batch_size=hp["batch_size"],
        buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        verbose=0,
        tensorboard_log=os.path.join(run_dir, "tb"),
        device=device,
    )
 
    # ── Callbacks ────────────────────────────────────────────
    metrics_cb = MetricsCallback()
 
    eval_cb = EvalCallback(
        eval_env_raw,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(timesteps // 10, 5_000),   # 10 evaluaciones durante el entrenamiento
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )
 
    # ── Entrenamiento ────────────────────────────────────────
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=[metrics_cb, eval_cb])
    elapsed = time.time() - t0
    
 
    # ── Evaluación final ─────────────────────────────────────
    mean_r, std_r = evaluate_policy(
        model, eval_env_raw,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        return_episode_rewards=False,
    )
 
    # ── Guardar modelo ───────────────────────────────────────
    model.save(os.path.join(run_dir, "model_final"))
    train_env.save(os.path.join(run_dir, "vecnorm.pkl"))
 
    # ── Curva de aprendizaje ─────────────────────────────────
    _save_learning_curve(metrics_cb, run_dir)
 
    train_env.close()
    eval_env_raw.close()
 
    metrics = {
        "mean_reward":  float(mean_r),
        "std_reward":   float(std_r),
        "elapsed_sec":  round(elapsed, 1),
        "n_episodes":   len(metrics_cb.episode_rewards),
        "final_ep_rew": float(np.mean(metrics_cb.episode_rewards[-20:])) if metrics_cb.episode_rewards else 0.0,
    }
    return metrics
 
 
def _save_learning_curve(cb: MetricsCallback, run_dir: str):
    if not cb.episode_rewards:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cb.timesteps_at_ep, cb.episode_rewards, alpha=0.4, linewidth=0.8, color="#90CAF9")
 
    # Suavizado con ventana deslizante
    if len(cb.episode_rewards) >= 10:
        window = max(10, len(cb.episode_rewards) // 20)
        smoothed = np.convolve(cb.episode_rewards, np.ones(window) / window, mode="valid")
        x_sm     = cb.timesteps_at_ep[window - 1:]
        ax.plot(x_sm, smoothed, linewidth=2, color="#1565C0", label="Smoothed")
 
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Learning curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "learning_curve.png"), dpi=100)
    plt.close(fig)
 
 
# ═══════════════════════════════════════════════════════════════
#  Grid expansion y naming
# ═══════════════════════════════════════════════════════════════
 
def expand_grid(grid: dict) -> list[dict]:
    """Devuelve todas las combinaciones posibles del grid."""
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, c)) for c in combos]
 
 
def hp_to_run_id(hp: dict, idx: int) -> str:
    """Nombre corto y legible para la carpeta de cada run."""
    short = {
        "learning_rate":    f"lr{hp['learning_rate']:.0e}",
        "batch_size":       f"bs{hp['batch_size']}",
        "gradient_steps":   f"gs{hp['gradient_steps']}",
        "cnn_output_dim":   f"cnn{hp['cnn_output_dim']}",
        "mlp_output_dim":   f"mlp{hp['mlp_output_dim']}",
        "frame_stack_k":    f"k{hp['frame_stack_k']}",
        "img_size":         f"img{hp['img_size']}",
        "curriculum_steps": f"cur{hp['curriculum_steps']//1000}k",
    }
    # Solo incluimos parámetros que varían en el grid
    varying = {k for k, v in PARAM_GRID.items() if len(v) > 1}
    parts   = [short[k] for k in short if k in varying]
    label   = "_".join(parts) if parts else "default"
    return f"run{idx:03d}_{label}"
 
 
# ═══════════════════════════════════════════════════════════════
#  Resumen final comparativo
# ═══════════════════════════════════════════════════════════════
 
def save_comparison(results: list[dict], out_dir: str):
    """Genera tabla JSON + figura comparativa de todas las runs."""
 
    # ── JSON ──────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "grid_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[summary] JSON → {summary_path}")
 
    # ── Tabla de texto ────────────────────────────────────────
    print("\n" + "═" * 80)
    print(f"{'Run':<30} {'Mean Reward':>12} {'Std':>8} {'Time (s)':>10}")
    print("─" * 80)
    sorted_res = sorted(results, key=lambda x: x["metrics"]["mean_reward"], reverse=True)
    for r in sorted_res:
        m = r["metrics"]
        print(f"{r['run_id']:<30} {m['mean_reward']:>12.2f} {m['std_reward']:>8.2f} {m['elapsed_sec']:>10.0f}s")
    print("═" * 80)
    print(f"\n🏆  Best run : {sorted_res[0]['run_id']}  (mean={sorted_res[0]['metrics']['mean_reward']:.2f})")
 
    # ── Figura ────────────────────────────────────────────────
    n    = len(results)
    ids  = [r["run_id"].split("_", 1)[1] or r["run_id"] for r in results]
    rew  = [r["metrics"]["mean_reward"] for r in results]
    stds = [r["metrics"]["std_reward"]  for r in results]
    colors = ["#FFD700" if i == np.argmax(rew) else "#42A5F5" for i in range(n)]
 
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    bars = ax.bar(range(n), rew, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Hyperparameter Grid Search — Final Evaluation")
    ax.axhline(max(rew), color="#FFD700", linestyle="--", linewidth=1,
               label=f"Best = {max(rew):.1f}")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "grid_comparison.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[summary] Comparison figure → {fig_path}")
 
    # ── Heatmaps para pares de hiperparámetros que varían ─────
    varying = [k for k, v in PARAM_GRID.items() if len(v) > 1]
    if len(varying) >= 2:
        for p1, p2 in itertools.combinations(varying[:4], 2):  # máx 6 pares
            _save_heatmap(results, p1, p2, out_dir)
 
 
def _save_heatmap(results, p1, p2, out_dir):
    v1 = sorted(set(r["hp"][p1] for r in results))
    v2 = sorted(set(r["hp"][p2] for r in results))
 
    mat = np.full((len(v2), len(v1)), np.nan)
    for r in results:
        i = v2.index(r["hp"][p2])
        j = v1.index(r["hp"][p1])
        # Si hay varias runs con estos valores, tomamos la media
        if np.isnan(mat[i, j]):
            mat[i, j] = r["metrics"]["mean_reward"]
        else:
            mat[i, j] = (mat[i, j] + r["metrics"]["mean_reward"]) / 2
 
    fig, ax = plt.subplots(figsize=(max(5, len(v1) * 1.5), max(4, len(v2) * 1.2)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(v1))); ax.set_xticklabels([str(v) for v in v1])
    ax.set_yticks(range(len(v2))); ax.set_yticklabels([str(v) for v in v2])
    ax.set_xlabel(p1); ax.set_ylabel(p2)
    ax.set_title(f"Mean Reward: {p1} vs {p2}")
    plt.colorbar(im, ax=ax)
 
    for i in range(len(v2)):
        for j in range(len(v1)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")
 
    plt.tight_layout()
    safe = lambda s: s.replace("/", "_")
    path = os.path.join(out_dir, f"heatmap_{safe(p1)}_vs_{safe(p2)}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[summary] Heatmap → {path}")
 
 
# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
 
def parse_args():
    parser = argparse.ArgumentParser(description="Grid search de hiperparámetros SAC Pusher")
    parser.add_argument("--out_dir",    type=str,  default="./grid_results",
                        help="Directorio raíz donde se guardan todas las runs")
    parser.add_argument("--timesteps",  type=int,  default=TIMESTEPS_PER_RUN,
                        help="Timesteps por run (default: %(default)s)")
    parser.add_argument("--device",     type=str,  default="cuda",
                        help="cuda | cpu")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Solo imprime las combinaciones sin entrenar")
    parser.add_argument("--resume",     type=str,  default=None,
                        help="Ruta a out_dir previo para saltar runs ya completadas")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
 
    combos = expand_grid(PARAM_GRID)
    total  = len(combos)
 
    print(f"\n{'═'*60}")
    print(f"  Hyperparameter Grid Search — SAC Pusher")
    print(f"  Combinaciones totales : {total}")
    print(f"  Timesteps por run     : {args.timesteps:,}")
    print(f"  Device                : {device}")
    print(f"  Output dir            : {args.out_dir}")
    print(f"{'═'*60}\n")
 
    # Imprimir tabla de combinaciones
    print(f"{'#':<5} " + "  ".join(f"{k:<18}" for k in PARAM_GRID.keys()))
    print("─" * (5 + 20 * len(PARAM_GRID)))
    for i, hp in enumerate(combos):
        row = f"{i:<5} " + "  ".join(f"{str(hp[k]):<18}" for k in PARAM_GRID.keys())
        print(row)
    print()
 
    if args.dry_run:
        print("[dry_run] Saliendo sin entrenar.")
        return
 
    # ── Cargar resultados previos si se hace resume ───────────
    completed_ids = set()
    all_results   = []
 
    if args.resume:
        summary_path = os.path.join(args.resume, "grid_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_results = json.load(f)
            completed_ids = {r["run_id"] for r in all_results}
            print(f"[resume] {len(completed_ids)} runs ya completadas, se saltarán.\n")
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    # ── Bucle principal ───────────────────────────────────────
    for idx, hp in enumerate(combos):
        run_id  = hp_to_run_id(hp, idx)
        run_dir = os.path.join(args.out_dir, run_id)
 
        if run_id in completed_ids:
            print(f"[{idx+1}/{total}] SKIP (ya completada): {run_id}")
            continue
 
        print(f"\n[{idx+1}/{total}] ▶  {run_id}")
        print("  Hiperparámetros:")
        for k, v in hp.items():
            marker = "  ← varía" if len(PARAM_GRID[k]) > 1 else ""
            print(f"    {k:<22} = {v}{marker}")
 
        try:
            metrics = train_one(
                hp=hp,
                run_dir=run_dir,
                timesteps=args.timesteps,
                device=device,
            )
        except Exception as e:
            print(f"  ⚠  Run fallida: {e}")
            metrics = {"mean_reward": float("-inf"), "std_reward": 0,
                       "elapsed_sec": 0, "n_episodes": 0, "final_ep_rew": 0,
                       "error": str(e)}
 
        result = {"run_id": run_id, "hp": hp, "metrics": metrics}
        all_results.append(result)
 
        # Guardar progreso parcial después de cada run
        partial_path = os.path.join(args.out_dir, "grid_summary.json")
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2)
 
        print(f"  ✓  mean_reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
              f"  ({metrics['elapsed_sec']:.0f}s)")
 
    # ── Resumen final ─────────────────────────────────────────
    if all_results:
        save_comparison(all_results, args.out_dir)
 
 
if __name__ == "__main__":
    main()