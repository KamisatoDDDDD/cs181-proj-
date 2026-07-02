#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting script for the CS181 2048-like project report.

Usage:
  python analyze_report_plots_v2.py --results-dir results --out-dir report_assets --seed-start 12 --num-games 180
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AGENT_ORDER = ["V0", "V1", "V2", "V3", "V4", "RL"]

AGENT_SHORT = {
    "V0": "Mono-C",
    "V1": "Full-C",
    "V2": "Basic-F",
    "V3": "Mono-F",
    "V4": "Full-F",
    "RL": "TD-Lin",
}

AGENT_PLOT = {
    "V0": "Mono-C\n(V0)",
    "V1": "Full-C\n(V1)",
    "V2": "Basic-F\n(V2)",
    "V3": "Mono-F\n(V3)",
    "V4": "Full-F\n(V4)",
    "RL": "TD-Lin\n(RL)",
}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def read_expectimax_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for v in range(5):
        agent = f"V{v}"
        files = sorted(results_dir.glob(f"final_v{v}_*.csv"))
        if not files:
            raise FileNotFoundError(f"No files found for {agent}: {results_dir / f'final_v{v}_*.csv'}")
        parts = []
        for f in files:
            df = pd.read_csv(f)
            df = normalize_columns(df)
            required = {"game_id", "score", "max_tile", "steps", "duration_sec"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{f} missing columns: {sorted(missing)}")
            tmp = pd.DataFrame({
                "agent": agent,
                "seed": df["game_id"].astype(int),
                "score": pd.to_numeric(df["score"], errors="raise"),
                "max_tile": pd.to_numeric(df["max_tile"], errors="raise"),
                "steps": pd.to_numeric(df["steps"], errors="raise"),
                "duration_sec": pd.to_numeric(df["duration_sec"], errors="raise"),
                "source_file": f.name,
            })
            parts.append(tmp)
        frames.append(pd.concat(parts, ignore_index=True))
    return pd.concat(frames, ignore_index=True)

def find_rl_file(results_dir: Path) -> Path:
    candidates = [
        results_dir / "eval_rl_games180.xlsx",
        results_dir / "eval_rl_games180.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted([p for p in results_dir.glob("*rl*.*") if p.suffix.lower() in {".csv", ".xlsx"}])
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find RL result file in results directory.")

def read_rl_results(results_dir: Path) -> pd.DataFrame:
    rl_file = find_rl_file(results_dir)
    if rl_file.suffix.lower() == ".xlsx":
        df = pd.read_excel(rl_file)
    else:
        df = pd.read_csv(rl_file)
    df = normalize_columns(df)

    if "duration_sec" not in df.columns and "time" in df.columns:
        df["duration_sec"] = df["time"]
    if "steps" not in df.columns and "step" in df.columns:
        df["steps"] = df["step"]

    required = {"seed", "score", "max_tile", "steps", "duration_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{rl_file} missing columns: {sorted(missing)}")

    return pd.DataFrame({
        "agent": "RL",
        "seed": df["seed"].astype(int),
        "score": pd.to_numeric(df["score"], errors="raise"),
        "max_tile": pd.to_numeric(df["max_tile"], errors="raise"),
        "steps": pd.to_numeric(df["steps"], errors="raise"),
        "duration_sec": pd.to_numeric(df["duration_sec"], errors="raise"),
        "source_file": rl_file.name,
    })

def load_all_results(results_dir: Path) -> pd.DataFrame:
    exp_df = read_expectimax_results(results_dir)
    rl_df = read_rl_results(results_dir)
    df = pd.concat([exp_df, rl_df], ignore_index=True)
    df["agent"] = pd.Categorical(df["agent"], categories=AGENT_ORDER, ordered=True)
    df = df.sort_values(["agent", "seed"]).reset_index(drop=True)
    df["runtime_per_step_sec"] = df["duration_sec"] / df["steps"]
    df["runtime_per_step_ms"] = 1000.0 * df["runtime_per_step_sec"]
    return df

def validate_seed_coverage(df: pd.DataFrame, seed_start: int, num_games: int) -> pd.DataFrame:
    expected = set(range(seed_start, seed_start + num_games))
    rows = []
    for agent in AGENT_ORDER:
        adf = df[df["agent"] == agent]
        seeds = set(adf["seed"].astype(int).tolist())
        rows.append({
            "agent": agent,
            "analysis_label": AGENT_SHORT[agent],
            "rows": len(adf),
            "unique_seeds": len(seeds),
            "expected_games": num_games,
            "num_missing": len(expected - seeds),
            "num_extra": len(seeds - expected),
            "duplicate_seed_rows": int(adf.duplicated(["seed"]).sum()),
        })
    qc = pd.DataFrame(rows)
    bad = qc[
        (qc["rows"] != num_games) |
        (qc["num_missing"] > 0) |
        (qc["num_extra"] > 0) |
        (qc["duplicate_seed_rows"] > 0)
    ]
    if bad.empty:
        print(f"[OK] All agents have {num_games} rows with seeds {seed_start}..{seed_start + num_games - 1}.")
    else:
        print("[WARNING] Seed coverage issues detected:")
        print(bad.to_string(index=False))
    return qc

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("agent", observed=False)
    summary = g.agg(
        games=("score", "count"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
        avg_max_tile=("max_tile", "mean"),
        median_max_tile=("max_tile", "median"),
        avg_steps=("steps", "mean"),
        median_steps=("steps", "median"),
        avg_duration_sec=("duration_sec", "mean"),
        median_duration_sec=("duration_sec", "median"),
        avg_runtime_per_step_ms=("runtime_per_step_ms", "mean"),
        median_runtime_per_step_ms=("runtime_per_step_ms", "median"),
    ).reset_index()
    rates = g.apply(lambda x: 100.0 * (x["max_tile"] >= 2048).mean(), include_groups=False)
    summary["reach_2048_pct"] = summary["agent"].map(rates.to_dict())
    summary["analysis_label"] = summary["agent"].map(AGENT_SHORT)
    summary["agent"] = pd.Categorical(summary["agent"], categories=AGENT_ORDER, ordered=True)
    summary = summary.sort_values("agent").reset_index(drop=True)
    return summary

def bucket_tile(tile: int) -> str:
    tile = int(tile)
    if tile <= 512:
        return r"$\leq$512"
    if tile == 1024:
        return "1024"
    if tile == 2048:
        return "2048"
    return r"$\geq$4096"

def save_score_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    data = [df[df["agent"] == a]["score"].to_numpy() for a in AGENT_ORDER]
    labels = [AGENT_PLOT[a] for a in AGENT_ORDER]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Final score (log scale)")
    ax.set_title("Score distribution across 180 fixed-seed games")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "score_boxplot_log.png", dpi=300)
    plt.close(fig)

def save_max_tile_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    tmp = df.copy()
    tmp["max_tile_bucket"] = tmp["max_tile"].apply(bucket_tile)
    bucket_order = [r"$\leq$512", "1024", "2048", r"$\geq$4096"]
    ctab = pd.crosstab(tmp["agent"], tmp["max_tile_bucket"], normalize="index") * 100.0
    ctab = ctab.reindex(index=AGENT_ORDER, columns=bucket_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bottom = np.zeros(len(ctab))
    x = np.arange(len(ctab.index))
    for b in bucket_order:
        vals = ctab[b].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=b)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_PLOT[a] for a in ctab.index.astype(str)], rotation=20, ha="right")
    ax.set_ylabel("Percentage of games (%)")
    ax.set_title("Distribution of maximum tile reached")
    ax.legend(title="Max tile bucket", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "max_tile_distribution_percent.png", dpi=300)
    plt.close(fig)
    ctab.to_csv(out_dir / "max_tile_distribution_bucketed.csv")

def save_tradeoff(summary: pd.DataFrame, out_dir: Path, mode: str = "game") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if mode == "game":
        x = summary["avg_duration_sec"].to_numpy()
        xlabel = "Average runtime per game (s, log scale)"
        title = "Performance-cost trade-off (per game)"
        filename = "score_runtime_tradeoff_game.png"
    else:
        x = summary["avg_runtime_per_step_ms"].to_numpy()
        xlabel = "Average runtime per step (ms, log scale)"
        title = "Performance-cost trade-off (per step)"
        filename = "score_runtime_tradeoff_step.png"
    y = summary["avg_score"].to_numpy()
    labels = summary["analysis_label"].tolist()
    ax.scatter(x, y, s=60)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)

def save_reach2048_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    agents = summary["agent"].astype(str).tolist()
    vals = summary["reach_2048_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(range(len(agents)), vals)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([AGENT_PLOT[a] for a in agents], rotation=20, ha="right")
    ax.set_ylabel("Reach-2048 rate (%)")
    ax.set_title("Percentage of games reaching at least 2048")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reach_2048_bar.png", dpi=300)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("report_assets"))
    parser.add_argument("--seed-start", type=int, default=12)
    parser.add_argument("--num-games", type=int, default=180)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_all_results(args.results_dir)
    qc = validate_seed_coverage(df, args.seed_start, args.num_games)
    summary = summarize(df)

    df.to_csv(args.out_dir / "merged_results.csv", index=False)
    qc.to_csv(args.out_dir / "seed_coverage_qc.csv", index=False)
    summary.to_csv(args.out_dir / "summary_for_plots.csv", index=False)

    save_score_boxplot(df, args.out_dir)
    save_max_tile_distribution(df, args.out_dir)
    save_tradeoff(summary, args.out_dir, mode="game")
    save_tradeoff(summary, args.out_dir, mode="step")
    save_reach2048_bar(summary, args.out_dir)

    show = summary[[
        "analysis_label", "games", "avg_score", "median_score", "max_score",
        "avg_max_tile", "avg_steps", "avg_duration_sec",
        "avg_runtime_per_step_ms", "reach_2048_pct"
    ]].copy()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print("\n=== Compact summary ===")
    print(show.to_string(index=False))
    print(f"\nSaved plots to: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
