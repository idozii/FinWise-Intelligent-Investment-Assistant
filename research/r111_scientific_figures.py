"""R11.1 paper-ready descriptive figures from VERIFIED offline R11 evidence.

No forecast model fitting, no external data download, no new R10 test.
Creates PNG and vector SVG plus SHA-256 manifest and captions outside
the original M5/R6/R8/R9/R11 evidence directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.r11_paper_evidence import compile_evidence, summarize_r8
from research.r9_selective_risk import THRESHOLDS, load_dev_input, policy_rows

PROTOCOL = "finwise-r112-readable-scientific-figures-v1"
M5_ASSETS = ("GLD", "IWM", "XLE", "XLV")
DEVELOPMENT_ASSETS = ("AAPL", "MSFT", "SPY", "TSLA")
METHODS = ("fixed_rolling", "delayed_feedback_aci_style")
R8_KEYS = ("asset", "model", "horizon_step", "calibration_method")

# Deterministic SVG element identifiers within a given Matplotlib version.
plt.rcParams["svg.hashsalt"] = "finwise-r112-figures-v1"


def sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def verified_m5_cells(registry: dict) -> list[dict]:
    if (registry.get("protocol") != "finwise-r11-manuscript-evidence-v1"
        or registry.get("evidence_stage")
            != "R11_DRAFT_MANUSCRIPT_HISTORICAL_AND_DEVELOPMENT_ONLY"
        or registry.get("independent_r10_evaluation_performed") is not False):
        raise ValueError("R11 registry must explicitly remain historical and R10-blocked")
    rows = registry.get("m5_first_run")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("All eight M5 first-run comparisons are required")
    seen = set()
    for row in rows:
        key = (row["asset"], row["horizon"])
        if key in seen or key[0] not in M5_ASSETS or key[1] not in (1, 7):
            raise ValueError("Duplicate or unauthorized M5 asset/horizon")
        seen.add(key)
        if type(row["n_origins"]) is not int or row["n_origins"] != 64:
            raise ValueError("M5 denominator is not the frozen 64-origin design")
        numbers = [row[k] for k in (
            "naive_mae", "ridge_mae", "ridge_minus_naive_mae",
            "paired_bootstrap_low", "paired_bootstrap_high",
            "ridge_ready_coverage",
        )]
        if not np.isfinite(numbers).all():
            raise ValueError("M5 figure input contains non-finite metrics")
        if not np.isclose(row["ridge_mae"] - row["naive_mae"],
                          row["ridge_minus_naive_mae"], atol=1e-8, rtol=1e-9):
            raise ValueError("M5 figure input MAE delta arithmetic mismatch")
        if not (row["paired_bootstrap_low"] <= row["paired_bootstrap_high"]
                and 0 <= row["ridge_ready_coverage"] <= 1
                and 0 <= row["ridge_n_accepted"] <= 64):
            raise ValueError("Invalid M5 bootstrap, coverage or acceptance fields")
    if seen != {(a, h) for a in M5_ASSETS for h in (1, 7)}:
        raise ValueError("Incomplete M5 asset/horizon registry")
    return sorted(rows, key=lambda r: (M5_ASSETS.index(r["asset"]), r["horizon"]))


def verify_r8_score_rows(records: pd.DataFrame, saved: pd.DataFrame,
                         registry: dict) -> pd.DataFrame:
    """Independently re-aggregate R8 READY rows; never trust a summary alone."""
    if len(saved) != 32 or saved.duplicated(list(R8_KEYS)).any():
        raise ValueError("R8 comparative summary must have 32 unique cells")
    if (set(saved.asset) != set(DEVELOPMENT_ASSETS)
        or set(saved.model) != {"naive", "ridge"}
        or set(saved.horizon_step) != {1, 7}
        or set(saved.calibration_method) != set(METHODS)):
        raise ValueError("R8 has missing or unauthorized experimental groups")
    expected = []
    for key, group in records.groupby(list(R8_KEYS), sort=True):
        ready = group.loc[group.interval_ready]
        if len(ready) == 0:
            raise ValueError("R8 requires at least one ready calibration origin")
        expected.append({
            **dict(zip(R8_KEYS, key)),
            "n_origins": len(group),
            "n_ready": len(ready),
            "n_accepted": int(group.accept_forecast.sum()),
            "coverage_ready": float((ready.covered == True).mean()),
            "mean_relative_width_ready": float(ready.relative_width.mean()),
        })
    derived = pd.DataFrame(expected)
    keys = list(R8_KEYS)
    merged = saved.merge(derived, on=keys, validate="one_to_one",
                         how="outer", indicator=True,
                         suffixes=("_saved", "_recomputed"))
    if len(merged) != 32 or not (merged._merge == "both").all():
        raise ValueError("R8 saved and verified group keys do not match")
    for column in ("n_origins", "n_ready", "n_accepted",
                   "coverage_ready", "mean_relative_width_ready"):
        if not np.allclose(merged[f"{column}_saved"],
                           merged[f"{column}_recomputed"],
                           rtol=1e-8, atol=1e-9):
            raise ValueError(f"R8 saved {column} changed from underlying intervals")
    if summarize_r8(saved) != registry.get("r8_development"):
        raise ValueError("R8 figures differ from verified R11 summary")
    return saved.sort_values(keys).reset_index(drop=True)


def verified_r9_curve(frame: pd.DataFrame) -> pd.DataFrame:
    curves, _ = policy_rows(frame)  # no model refit or external test access
    if (len(curves) != 160 or
        set(curves.asset) != set(DEVELOPMENT_ASSETS) or
        set(curves.max_relative_width) != set(THRESHOLDS)):
        raise ValueError("R9 policy figure has an incomplete development design")
    if (curves.loc[curves.n_accepted == 0,
                   ["relative_mae_selected", "interval_coverage_selected"]]
            .notna().any().any()):
        raise ValueError("R9 zero-acceptance risk is undefined, never zero")
    return curves


def _save(fig, output: Path, stem: str) -> dict:
    produced = {}
    for ext in ("png", "svg"):
        final = output / f"{stem}.{ext}"
        candidate = output / f".{stem}.{ext}.temporary"
        fig.savefig(candidate, format=ext, dpi=180, bbox_inches="tight",
                    facecolor="white")
        generated = candidate.read_bytes()
        if final.exists() and final.read_bytes() != generated:
            candidate.unlink()
            raise FileExistsError(
                f"Existing figure changed at {final}; version the output folder"
            )
        candidate.replace(final)
        produced[final.name] = sha256(final)
    plt.close(fig)
    return produced


def plot_m5(cells: list[dict], output: Path) -> dict:
    labels = [f"{r['asset']} · H{r['horizon']}" for r in cells]
    means = np.array([r["ridge_minus_naive_mae"] for r in cells], float)
    lows = np.array([r["paired_bootstrap_low"] for r in cells], float)
    highs = np.array([r["paired_bootstrap_high"] for r in cells], float)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(cells))
    ax.errorbar(means, y, xerr=[means-lows, highs-means], fmt="o",
                linewidth=1.6, capsize=3, label="Ridge − Naive MAE (95% descriptive CI)")
    ax.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ridge − Naive MAE (raw Close price units)")
    ax.set_title("M5-v2: all eight historical comparisons")
    ax.grid(axis="x", alpha=.2)
    fig.text(.12, .015, "64 origins per cell · moving-block bootstrap intervals are descriptive, unadjusted. "
             "XLE H7 raw Close crosses a known split.", fontsize=8)
    return _save(fig, output, "fig01_m5_paired_mae")


def plot_r8(scores: pd.DataFrame, model: str, output: Path) -> dict:
    """Side-by-side horizons; asset colors and method markers, no crowded labels."""
    group = scores.loc[scores.model == model].copy()
    if len(group) != 16:
        raise ValueError("R8 plotting requires all 8 asset/horizon pairs and two methods")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
    palette = {asset: plt.get_cmap("tab10")(i)
               for i, asset in enumerate(DEVELOPMENT_ASSETS)}
    for horizon, ax in zip((1, 7), axes):
        for asset in DEVELOPMENT_ASSETS:
            pair = group.loc[(group.asset == asset) &
                             (group.horizon_step == horizon)]
            if len(pair) != 2 or set(pair.calibration_method) != set(METHODS):
                raise ValueError("R8 requires fixed and ACI-style for each asset/horizon")
            pair = pair.set_index("calibration_method")
            fixed, adaptive = pair.loc[METHODS[0]], pair.loc[METHODS[1]]
            x = [100*fixed.mean_relative_width_ready,
                 100*adaptive.mean_relative_width_ready]
            y = [100*fixed.coverage_ready, 100*adaptive.coverage_ready]
            ax.plot(x, y, linewidth=1.3, alpha=.7, color=palette[asset],
                    zorder=2)
            ax.scatter([x[0]], [y[0]], s=64, marker="o",
                       color=palette[asset], zorder=4)
            ax.scatter([x[1]], [y[1]], s=75, marker="x", linewidths=2,
                       color=palette[asset], zorder=5)
        ax.axhline(90, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"H{horizon} · {horizon}-session target", pad=10)
        ax.set_xlabel("Mean interval width / last Close (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(75, 97)
        ax.grid(alpha=.17)
    axes[0].set_ylabel("Ready-origin interval coverage (%)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=palette[asset], linewidth=2, label=asset)
        for asset in DEVELOPMENT_ASSETS
    ] + [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=7, label="Fixed rolling"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None",
               markersize=8, label="Delayed-feedback ACI-style"),
    ]
    fig.suptitle(f"R8: observed coverage versus interval width · {model.title()} · DEV ONLY",
                 fontsize=15, y=.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91),
               ncol=6, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.23, wspace=.16)
    fig.text(.5, .115,
             "Connecting segments compare the same asset/horizon and identical "
             "point forecasts under two calibration methods.",
             fontsize=9, ha="center")
    fig.text(.5, .075,
             "Exploratory development only; y-axis zoomed to 75–97%. "
             "Nominal 90% coverage is not a guarantee.",
             fontsize=9, ha="center")
    return _save(fig, output, f"fig02_r8_{model}_coverage_width")


def r9_policy_table(view: pd.DataFrame) -> list[list[str]]:
    """Exact five-threshold summary: duplicate points and missing risks visible."""
    if len(view) != 10 or view.duplicated(
        ["calibration_method", "max_relative_width"]
    ).any():
        raise ValueError("R9 requires exactly two methods by five thresholds")
    rows = []
    for threshold in THRESHOLDS:
        line = [f"{100*threshold:.0f}%"]
        for method in METHODS:
            match = view.loc[
                (view.calibration_method == method) &
                (np.isclose(view.max_relative_width, threshold))
            ]
            if len(match) != 1:
                raise ValueError("R9 missing a method/threshold policy cell")
            cell = match.iloc[0]
            n, all_origins = int(cell.n_accepted), int(cell.n_origins)
            if n < 0 or all_origins <= 0 or n > all_origins:
                raise ValueError("Invalid R9 accepted-origins denominator")
            if n == 0:
                if not pd.isna(cell.relative_mae_selected):
                    raise ValueError("Zero accepted policy must have missing selected risk")
                risk = "N/A"
            else:
                if not np.isfinite(float(cell.relative_mae_selected)):
                    raise ValueError("Non-finite selected risk")
                risk = f"{100*cell.relative_mae_selected:.2f}%"
                if n < 30:
                    risk += " *"
            line.extend([f"{n}/{all_origins}", risk])
        rows.append(line)
    return rows


def plot_r9(curve: pd.DataFrame, asset: str, output: Path) -> dict:
    """Discrete risk/availability points + exact policy table, no spurious lines."""
    view = curve.loc[(curve.asset == asset) &
                     (curve.model == "ridge") &
                     (curve.horizon_step == 7)].copy()
    table_rows = r9_policy_table(view)
    fig, (ax, table_ax) = plt.subplots(
        1, 2, figsize=(12.4, 5.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    methods = (
        ("fixed_rolling", "Fixed rolling", "o"),
        ("delayed_feedback_aci_style", "Delayed-feedback ACI-style", "x"),
    )
    valid_all = []
    for method, label, marker in methods:
        sub = view.loc[view.calibration_method == method]
        valid = sub.loc[(sub.n_accepted > 0) &
                        sub.relative_mae_selected.notna()]
        valid_all.extend((100*valid.relative_mae_selected).tolist())
        if len(valid):
            ax.scatter(100*valid.acceptance_rate_all,
                       100*valid.relative_mae_selected,
                       s=70, marker=marker, linewidths=2,
                       label=label, zorder=3)
        else:
            ax.scatter([], [], marker=marker, label=label + " (no accepted)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(1., max(valid_all, default=0)*1.15))
    ax.set_xlabel("Accepted / all forecast origins (%)")
    ax.set_ylabel("Selected relative MAE (%)")
    ax.set_title(f"Ridge H7 · {asset} · development only", pad=14)
    ax.grid(alpha=.20)
    ax.legend(fontsize=9, loc="upper right", framealpha=.9)

    # The table lists all policy thresholds, including repeated
    # acceptance sets and zero-acceptance cells that have no plot point.
    table_ax.axis("off")
    table_ax.set_title("Exact discrete policy outcomes", fontsize=12, pad=14)
    table = table_ax.table(
        cellText=table_rows,
        colLabels=("Width", "Fixed n", "Fixed rMAE",
                   "ACI n", "ACI rMAE"),
        cellLoc="center", colLoc="center",
        loc="center", bbox=[0, .13, 1, .77]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eff3f2")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f9f9")

    fig.suptitle(f"R9: selective forecast risk–availability · {asset} · DEV ONLY",
                 fontsize=15, y=.99)
    fig.subplots_adjust(left=.085, right=.99, top=.82, bottom=.23, wspace=.18)
    fig.text(.5, .12,
             "Five discrete thresholds; repeated points are NOT separate observations. "
             "N/A = no accepted forecasts; * denotes n < 30.",
             fontsize=9, ha="center")
    fig.text(.5, .075,
             "Y-axis starts at zero. Policies select different origins; "
             "selected MAE does not establish causal improvement.",
             fontsize=9, ha="center")
    return _save(fig, output, f"fig03_r9_{asset.lower()}_risk_availability")


def generate(registry: dict, r8_saved: pd.DataFrame,
             r8_intervals: pd.DataFrame, r9: pd.DataFrame,
             out_dir: Path) -> dict:
    cells = verified_m5_cells(registry)
    r8 = verify_r8_score_rows(r8_intervals, r8_saved, registry)
    curves = verified_r9_curve(r8_intervals)
    # Full numerical reconciliation: a tampered saved R9 CSV cannot be
    # rendered, even if its narrower manuscript-summary fields still match.
    group_keys = [
        "asset", "model", "horizon_step", "calibration_method",
        "max_relative_width",
    ]
    if len(r9) != 160 or r9.duplicated(group_keys).any():
        raise ValueError("Incomplete or duplicate saved R9 development grid")
    joined = r9.merge(curves, on=group_keys, how="outer",
                      indicator=True, validate="one_to_one",
                      suffixes=("_saved", "_recomputed"))
    if len(joined) != 160 or not (joined._merge == "both").all():
        raise ValueError("R9 saved policy keys differ from verified source")
    numeric_checks = (
        "n_origins", "n_ready", "n_accepted", "acceptance_rate_all",
        "acceptance_rate_ready", "interval_coverage_ready",
        "interval_coverage_selected", "mae_selected",
        "relative_mae_selected", "mean_relative_interval_width_selected",
    )
    for name in numeric_checks:
        if not np.allclose(
            joined[name + "_saved"], joined[name + "_recomputed"],
            equal_nan=True, rtol=1e-8, atol=1e-9
        ):
            raise ValueError(f"Saved R9 {name} differs from verified development rows")
    if not (joined.selection_warning_saved ==
            joined.selection_warning_recomputed).all():
        raise ValueError("R9 selected-sample warnings were changed")
    # Reconcile R9 summary used in the manuscript to the exact recomputed 10% curve.
    selected = curves.loc[(curves.model == "ridge") &
                          (curves.horizon_step == 7) &
                          np.isclose(curves.max_relative_width, .10)]
    for paper_row in registry["r9_development_ridge_h7_width_10pct"]:
        matched = selected.loc[
            (selected.asset == paper_row["asset"]) &
            (selected.calibration_method == paper_row["method"])
        ]
        if len(matched) != 1 or int(matched.iloc[0].n_accepted) != paper_row["n_accepted"]:
            raise ValueError("R9 paper figures disagree with R11 registry")
    # Do not create a false impression of a single model-ready independent test.
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    hashes.update(plot_m5(cells, out_dir))
    for model in ("naive", "ridge"):
        hashes.update(plot_r8(r8, model, out_dir))
    for asset in DEVELOPMENT_ASSETS:
        hashes.update(plot_r9(curves, asset, out_dir))
    return {
        "protocol": PROTOCOL,
        "status": "VERIFIED_DESCRIPTIVE_HISTORICAL_AND_DEVELOPMENT_FIGURES",
        "r10_independent_evaluation_performed": False,
        "n_figures": 7,
        "assets_m5": list(M5_ASSETS),
        "assets_dev": list(DEVELOPMENT_ASSETS),
        "figure_sha256": hashes,
        "interpretation": [
            "M5 confidence intervals descriptive, no multiple-testing correction",
            "M5 original XLE split event and raw-close target remain unchanged",
            "R8 and R9 are exploratory development comparisons, not independent validation",
            "Selected forecast subsets differ across policies",
            "No accepted forecasts means unavailable selected loss, not zero risk",
            "No live coverage, profitability, causal or investment advice claims",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m5-dir", type=Path,
                   default=Path("reports/backtest/m5_tiingo_v2"))
    p.add_argument("--r6-manifest", type=Path,
                   default=Path("reports/backtest/r6_corporate_actions/r6_audit_manifest.json"))
    p.add_argument("--r8-dir", type=Path,
                   default=Path("reports/backtest/r8_development"))
    p.add_argument("--r9-dir", type=Path,
                   default=Path("reports/backtest/r9_risk_coverage_dev"))
    p.add_argument("--dev-predictions", type=Path,
                   default=Path("reports/backtest/m21_multiasset/asset_predictions.csv"))
    p.add_argument("--r11-registry", type=Path,
                   default=Path("reports/backtest/r11_manuscript_evidence/r11_evidence_registry.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("reports/backtest/r112_publication_figures"))
    args = p.parse_args(argv)
    sources = (args.m5_dir, args.r8_dir, args.r9_dir,
               args.r11_registry.parent)
    if any(args.output_dir.resolve().is_relative_to(d.resolve()) for d in sources):
        raise ValueError("Figures may not overwrite any original evidence directory")
    # Expensive but deliberate: this verifies the frozen first run and original
    # development evidence every time, rather than just trusting a JSON file.
    regenerated = compile_evidence(
        args.m5_dir, args.r8_dir, args.r9_dir,
        args.dev_predictions, args.r6_manifest
    )
    saved = json.loads(args.r11_registry.read_text(encoding="utf-8"))
    if regenerated != saved:
        raise ValueError("R11 registry does not match original audited evidence")
    intervals = load_dev_input(
        args.r8_dir/"r8_intervals_development.csv",
        args.r8_dir/"r8_manifest.json", args.dev_predictions
    )
    saved_scores = pd.read_csv(args.r8_dir/"r8_comparison_development.csv")
    figure_manifest = generate(
        regenerated, saved_scores, intervals,
        pd.read_csv(args.r9_dir/"r9_risk_coverage_dev.csv"),
        args.output_dir
    )
    figure_manifest["source_sha256"] = {
        "r11_registry": sha256(args.r11_registry),
        "r8_intervals": sha256(args.r8_dir/"r8_intervals_development.csv"),
        "r8_scores": sha256(args.r8_dir/"r8_comparison_development.csv"),
        "r9_policy_grid": sha256(args.r9_dir/"r9_risk_coverage_dev.csv"),
    }
    manifest = args.output_dir/"r111_figures_manifest.json"
    content = json.dumps(figure_manifest, indent=2, allow_nan=False) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise FileExistsError("Previous figure manifest differs; use a versioned output directory")
    manifest.write_text(content, encoding="utf-8")
    print("R11.2 FIGURES VERIFIED: 7 readable plots, 14 PNG/SVG files")
    print("NO new independent R10 test; no live coverage or investment claims")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
