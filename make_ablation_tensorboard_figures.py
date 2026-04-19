from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar_from_event_file(event_file: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    acc = EventAccumulator(
        event_file,
        size_guidance={"scalars": 0},
    )
    acc.Reload()

    available_tags = acc.Tags().get("scalars", [])
    if tag not in available_tags:
        raise ValueError(f"Tag '{tag}' no encontrada en {event_file}. Tags disponibles: {available_tags}")

    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def find_event_file(tb_dir: str) -> str:
    tb_path = Path(tb_dir)
    files = sorted(tb_path.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No se encontró ningún event file dentro de {tb_dir}")
    return str(files[0])


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window or window <= 1:
        return y.copy()
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    pad_left = window - 1
    if pad_left > 0:
        prefix = np.full(pad_left, y_smooth[0])
        y_smooth = np.concatenate([prefix, y_smooth])
    return y_smooth


def plot_comparison(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    title: str,
    ylabel: str,
    smooth_window: int = 1,
):
    plt.figure(figsize=(9, 5))

    colors = {
        "multimodal_curriculum": "#1f4e79",
        "multimodal_no_curriculum": "#17becf",
        "features_only": "#e83e8c",
        "images_only": "#f0a500",
    }

    for name, (x, y) in series.items():
        y_plot = moving_average(y, smooth_window) if smooth_window > 1 else y
        plt.plot(x, y_plot, label=name, linewidth=2.2, color=colors.get(name, None))

    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Genera figuras comparativas desde logs de TensorBoard de la ablación.")
    parser.add_argument("--ablation_dir", type=str, required=True,
                        help="Directorio raíz de la ablación, p.ej. ./ablation_results_common_300k")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directorio donde guardar las figuras")
    parser.add_argument("--smooth_eval", type=int, default=1,
                        help="Ventana de suavizado para eval/mean_reward")
    parser.add_argument("--smooth_rollout", type=int, default=1,
                        help="Ventana de suavizado para rollout/ep_rew_mean")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    variants = [
        "multimodal_curriculum",
        "multimodal_no_curriculum",
        "features_only",
        "images_only",
    ]

    eval_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rollout_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for variant in variants:
        tb_dir = os.path.join(args.ablation_dir, variant, "tb")
        event_file = find_event_file(tb_dir)

        x_eval, y_eval = load_scalar_from_event_file(event_file, "eval/mean_reward")
        x_roll, y_roll = load_scalar_from_event_file(event_file, "rollout/ep_rew_mean")

        eval_series[variant] = (x_eval, y_eval)
        rollout_series[variant] = (x_roll, y_roll)

    plot_comparison(
        eval_series,
        os.path.join(args.out_dir, "ablation_eval_mean_reward.png"),
        title="Ablation study - evaluation reward",
        ylabel="Mean eval reward",
        smooth_window=args.smooth_eval,
    )

    plot_comparison(
        rollout_series,
        os.path.join(args.out_dir, "ablation_rollout_ep_rew_mean.png"),
        title="Ablation study - training reward",
        ylabel="Mean episode reward",
        smooth_window=args.smooth_rollout,
    )
    
    plot_combined(
        eval_series,
        rollout_series,
        os.path.join(args.out_dir, "ablation_combined.png"),
        smooth_eval=args.smooth_eval,
        smooth_rollout=args.smooth_rollout,
    )

def plot_combined(
    eval_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rollout_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    smooth_eval: int = 1,
    smooth_rollout: int = 1,
):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    colors = {
        "multimodal_curriculum": "#1f4e79",
        "multimodal_no_curriculum": "#17becf",
        "features_only": "#e83e8c",
        "images_only": "#f0a500",
    }

    # --- TOP: EVAL ---
    ax = axes[0]
    for name, (x, y) in eval_series.items():
        y_plot = moving_average(y, smooth_eval) if smooth_eval > 1 else y
        ax.plot(x, y_plot, label=name, linewidth=2.2, color=colors.get(name, None))

    ax.set_ylabel("Mean eval reward")
    ax.set_title("Ablation study - evaluation reward")
    ax.legend()
    ax.grid(alpha=0.2)

    # --- BOTTOM: ROLLOUT ---
    ax = axes[1]
    for name, (x, y) in rollout_series.items():
        y_plot = moving_average(y, smooth_rollout) if smooth_rollout > 1 else y
        ax.plot(x, y_plot, label=name, linewidth=2.2, color=colors.get(name, None))

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Ablation study - training reward")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[saved] {out_path}")
    
if __name__ == "__main__":
    main()