# make_report_figures.py
from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_grid_summary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_rollout_frames(gif_path: str, out_path: str, n_frames: int = 6):
    frames = imageio.mimread(gif_path)
    if not frames:
        raise RuntimeError(f"No frames found in {gif_path}")

    n = min(n_frames, len(frames))
    idxs = [round(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
    chosen = [frames[i] for i in idxs]

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    if n == 1:
        axes = [axes]

    for ax, idx, fr in zip(axes, idxs, chosen):
        ax.imshow(fr)
        ax.set_title(f"t={idx}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_cost_by_resolution(results, out_path: str):
    by_res = defaultdict(list)
    by_res_reward = defaultdict(list)

    for r in results:
        img = r["hp"]["img_size"]
        by_res[img].append(r["metrics"]["elapsed_sec"] / 3600.0)
        by_res_reward[img].append(r["metrics"]["mean_reward"])

    labels = sorted(by_res.keys())
    means = [np.mean(by_res[k]) for k in labels]
    stds = [np.std(by_res[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
           color=["#42A5F5", "#EF5350"][:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{k}x{k}" for k in labels])
    ax.set_ylabel("Tiempo de entrenamiento (horas)")
    ax.set_title("Coste computacional por resolución")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[saved] {out_path}")


def pick_best_worst_stable(results):
    best = max(results, key=lambda r: r["metrics"]["mean_reward"])
    worst = min(results, key=lambda r: r["metrics"]["mean_reward"])

    positive = [r for r in results if r["metrics"]["mean_reward"] > 0]
    pool = positive if positive else results
    stable = min(pool, key=lambda r: r["metrics"]["std_reward"])
    return best, worst, stable


def make_latex_tables(results, out_path: str):
    by_res = defaultdict(list)
    by_res_reward = defaultdict(list)
    for r in results:
        img = r["hp"]["img_size"]
        by_res[img].append(r["metrics"]["elapsed_sec"] / 3600.0)
        by_res_reward[img].append(r["metrics"]["mean_reward"])

    best, worst, stable = pick_best_worst_stable(results)

    lines = []

    lines += [
        "% ---- Cost by resolution ----",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Coste computacional medio por resolución de imagen.}",
        "\\label{tab:cost_by_resolution}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Resolución & Tiempo medio (h) & Reward medio \\\\",
        "\\midrule",
    ]
    for img in sorted(by_res.keys()):
        mt = np.mean(by_res[img])
        st = np.std(by_res[img])
        mr = np.mean(by_res_reward[img])
        lines.append(f"${img}\\times{img}$ & {mt:.2f} $\\pm$ {st:.2f} & {mr:.1f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% ---- Best / worst / stable ----",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Resumen de configuraciones destacadas del grid search.}",
        "\\label{tab:grid_best_worst_stable}",
        "\\begin{tabular}{p{2.0cm}p{4.0cm}}",
        "\\toprule",
        "Categoría & Configuración \\\\",
        "\\midrule",
        f"Mejor run & {best['run_id'].replace('_', '\\_')} "
        f"(reward={best['metrics']['mean_reward']:.1f}, std={best['metrics']['std_reward']:.1f}) \\\\",
        f"Peor run & {worst['run_id'].replace('_', '\\_')} "
        f"(reward={worst['metrics']['mean_reward']:.1f}, std={worst['metrics']['std_reward']:.1f}) \\\\",
        f"Más estable & {stable['run_id'].replace('_', '\\_')} "
        f"(reward={stable['metrics']['mean_reward']:.1f}, std={stable['metrics']['std_reward']:.1f}) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report assets from existing results")
    parser.add_argument("--grid_summary", type=str, default="./grid_results/grid_summary.json")
    parser.add_argument("--rollout_gif", type=str, default="./eval_results/ep00_rollout.gif")
    parser.add_argument("--out_dir", type=str, default="./report_assets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = load_grid_summary(args.grid_summary)

    save_rollout_frames(
        gif_path=args.rollout_gif,
        out_path=os.path.join(args.out_dir, "rollout_frames.png"),
        n_frames=6,
    )

    save_cost_by_resolution(
        results,
        out_path=os.path.join(args.out_dir, "cost_by_resolution.png"),
    )

    make_latex_tables(
        results,
        out_path=os.path.join(args.out_dir, "report_tables.tex"),
    )


if __name__ == "__main__":
    main()