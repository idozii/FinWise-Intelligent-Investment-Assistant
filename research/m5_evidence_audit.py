"""M5.1 immutable *post-evaluation* evidence audit, no model retraining.

Checks the existing first M5-v2 evaluation's file hashes, scores, selection,
interval coverage and bootstrap intervals against origin-level predictions.
Flags extreme errors descriptively but NEVER changes any score or sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.calibration_audit import (
    matched_selection_audit, model_audit, validated_rows,
)
from research.frozen_eval_tiingo import (
    LOCKED_V2_SHA256, load_v2_protocol,
)
from research.robustness import paired_block_bootstrap
from scripts.m5_download_tiingo import verify_snapshot

PROTOCOL = "finwise-m51-first-evaluation-evidence-audit-v1"
FILENAME_MAP = {
    "pred": "holdout_predictions.csv",
    "interval": "holdout_interval_predictions.csv",
    "metrics": "holdout_metrics.csv",
    "paired": "holdout_paired_comparisons.csv",
    "calibration": "holdout_calibration.csv",
    "selected": "holdout_matched_accepted.csv",
    "preflight": "preflight_manifest.json",
    "result": "holdout_result_manifest.json",
}
GROUP = ["asset", "model", "horizon_step"]
PAIRED_GROUP = ["asset", "horizon_step", "model"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_same_numeric(reference: pd.DataFrame, reported: pd.DataFrame,
                         key: list[str], cols: list[str], name: str) -> None:
    if (reference.duplicated(key).any() or reported.duplicated(key).any()
            or len(reference) != len(reported)):
        raise ValueError(f"{name}: missing or duplicate comparison keys")
    common = reference.merge(
        reported, on=key, how="outer", indicator=True, validate="one_to_one",
        suffixes=("_expected", "_observed"),
    )
    if (common["_merge"] != "both").any():
        raise ValueError(f"{name}: missing/extra rows")
    for col in cols:
        exp = pd.to_numeric(common[f"{col}_expected"], errors="raise")
        obs = pd.to_numeric(common[f"{col}_observed"], errors="raise")
        if not np.allclose(exp, obs, atol=1e-8, rtol=1e-9, equal_nan=True):
            raise ValueError(f"{name}: inconsistent {col} compared with origin-level data")


def verify_first_run(directory: Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Hard stop if source snapshot, protocol, completeness or results drift."""
    paths = {name: directory / file for name, file in FILENAME_MAP.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing first-run artifact: {name}: {path}")
    original_preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if (original_preflight.get("protocol_sha256") != LOCKED_V2_SHA256
            or result.get("protocol_sha256") != LOCKED_V2_SHA256
            or result.get("protocol") != spec["protocol"]):
        raise ValueError("First run is not tied to locked Tiingo M5-v2 protocol")
    if result.get("stage") != "FIRST_EXTERNAL_ASSET_RETROSPECTIVE_EVALUATION":
        raise ValueError("Expected the original first-evaluation result")
    sources = original_preflight.get("sources", [])
    if set(s["asset"] for s in sources) != set(spec["reserved_external_assets"]):
        raise ValueError("Preflight must contain exactly the frozen four assets")
    for item in sources:
        asset = item["asset"]
        path = Path(item["file"])
        if not path.is_file() or sha(path) != item["source_sha256"]:
            raise ValueError(f"{asset}: local source file was moved, removed or modified")
        verify_snapshot(asset, path)
        if result["preflight_sources_sha256"].get(asset) != item["source_sha256"]:
            raise ValueError(f"{asset}: evaluation source hash differs from preflight")

    predictions = pd.read_csv(paths["pred"])
    validated_rows(paths["pred"], paths["interval"])
    if len(predictions) != result["n_rows"]:
        raise ValueError("Result manifest's n_rows differs from prediction file")
    if set(predictions.asset) != set(spec["reserved_external_assets"]):
        raise ValueError("Evaluated assets do not match frozen protocol")
    if set(predictions.model) != set(spec["models"]):
        raise ValueError("Evaluated models do not match frozen protocol")
    if set(predictions.horizon_step) != set(spec["horizons"]):
        raise ValueError("Evaluated horizons do not match frozen protocol")
    dates = pd.to_datetime(predictions.train_end, utc=True, errors="raise")
    targets = pd.to_datetime(predictions.target_date, utc=True, errors="raise")
    start = pd.Timestamp(spec["forecast_origin_start"], tz="UTC")
    end = pd.Timestamp(spec["forecast_origin_end"], tz="UTC")
    if ((dates < start) | (dates > end) | (targets > end)
            | (targets <= dates)).any():
        raise ValueError("Forecast origins/targets violate frozen evaluation window")
    counts = predictions.groupby(GROUP).origin_index.nunique()
    if len(counts) != len(spec["reserved_external_assets"]) * len(spec["models"]) * len(spec["horizons"]):
        raise ValueError("Missing one or more asset/model/horizon groups")
    if counts.min() != counts.max():
        raise ValueError("Unbalanced origin count across frozen asset/model/horizon")
    return predictions, result


def recalculated_diagnostics(directory: Path, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent arithmetic checks from existing scores, not new fitting."""
    paths = {name: directory / file for name, file in FILENAME_MAP.items()}
    predictions, _ = verify_first_run(directory, spec)
    pairs = []
    metric_rows = []
    b = spec["bootstrap"]
    for (asset, model, horizon), group in predictions.groupby(GROUP, sort=True):
        error = np.abs(group.actual.to_numpy(float) - group.predicted.to_numpy(float))
        metric_rows.append({
            "asset": asset, "model": model, "horizon_step": horizon,
            "n_origins": group.origin_index.nunique(),
            "mae": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        })
    _assert_same_numeric(
        pd.DataFrame(metric_rows), pd.read_csv(paths["metrics"]), GROUP,
        ["n_origins", "mae", "rmse"], "holdout_metrics.csv"
    )
    for asset in spec["reserved_external_assets"]:
        for h in spec["horizons"]:
            group = predictions[(predictions.asset == asset)
                                & (predictions.horizon_step == h)]
            diff, low, high = paired_block_bootstrap(
                group, block_length=b["block_length_origins"],
                iterations=b["iterations"], seed=b["seed"],
            )
            pairs.append({
                "asset": asset, "model": "ridge", "horizon_step": h,
                "n_origins": group.origin_index.nunique(),
                "mae_delta_model_minus_naive": diff,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            })
    _assert_same_numeric(
        pd.DataFrame(pairs), pd.read_csv(paths["paired"]), PAIRED_GROUP,
        ["n_origins", "mae_delta_model_minus_naive",
         "bootstrap_ci_low", "bootstrap_ci_high"],
        "holdout_paired_comparisons.csv",
    )
    validated = validated_rows(paths["pred"], paths["interval"])
    cal = model_audit(validated, spec["intervals"]["alpha"])
    _assert_same_numeric(
        cal, pd.read_csv(paths["calibration"]), GROUP,
        ["n_total", "n_ready", "n_accepted", "acceptance_rate_all",
         "acceptance_rate_ready", "coverage_ready", "coverage_selected",
         "mae_selected", "mean_relative_width_ready"], "holdout_calibration.csv",
    )
    fair = matched_selection_audit(validated)
    _assert_same_numeric(
        fair, pd.read_csv(paths["selected"]), PAIRED_GROUP,
        ["n_origins", "n_both_calibrated", "n_both_accepted",
         "naive_mae_same_accepted", "model_mae_same_accepted",
         "delta_mae_model_minus_naive_same_accepted"],
        "holdout_matched_accepted.csv",
    )
    return pd.DataFrame(pairs), cal


def descriptive_error_flags(predictions: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc flags only: ALL observations remain in every primary score."""
    rows = predictions.copy()
    rows["abs_error"] = (rows.actual - rows.predicted).abs()
    rows["realized_abs_return_since_origin"] = (
        rows.actual / rows.last_observed_close - 1.0
    ).abs()
    rows["large_abs_return_gt_25pct"] = (
        rows.realized_abs_return_since_origin > 0.25
    )
    rows = rows.sort_values(
        ["asset", "model", "horizon_step", "abs_error"],
        ascending=[True, True, True, False]
    )
    # Top 3 errors per asset/horizon/model only, not a cherry-picked subset
    # for score recalculation.
    return rows.groupby(GROUP, sort=True).head(3)[
        GROUP + ["origin_index", "train_end", "target_date", "abs_error",
                 "realized_abs_return_since_origin",
                 "large_abs_return_gt_25pct"]
    ].reset_index(drop=True)


def generate_report(
    comparisons: pd.DataFrame, calibration: pd.DataFrame,
    selected: pd.DataFrame, predictions: pd.DataFrame,
) -> str:
    lines = [
        "# M5-v2 — First external-asset evaluation: verified descriptive audit",
        "",
        "**Stage:** first historical external-asset replication; post-hoc evidence audit.",
        "All model hyperparameters and date/asset choices are unchanged.",
        "Not a prospective future test, coverage guarantee or trading evaluation.",
        "",
        "| Asset | Horizon | Ridge−Naive MAE | Bootstrap 95% CI | Ridge coverage | Ridge accepted | Both accepted | Selected ΔMAE |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparisons.sort_values(["asset", "horizon_step"]).itertuples(index=False):
        c = calibration[
            (calibration.asset == row.asset) &
            (calibration.model == "ridge") &
            (calibration.horizon_step == row.horizon_step)
        ].iloc[0]
        s = selected[
            (selected.asset == row.asset) &
            (selected.horizon_step == row.horizon_step) &
            (selected.model == "ridge")
        ].iloc[0]
        sd = (f"{s.delta_mae_model_minus_naive_same_accepted:+.3f}"
              if pd.notna(s.delta_mae_model_minus_naive_same_accepted)
              else "N/A")
        lines.append(
            f"| {row.asset} | {row.horizon_step} | "
            f"{row.mae_delta_model_minus_naive:+.3f} | "
            f"[{row.bootstrap_ci_low:+.3f}, {row.bootstrap_ci_high:+.3f}] | "
            f"{c.coverage_ready:.1%} | {int(c.n_accepted)}/{int(c.n_total)} | "
            f"{int(s.n_both_accepted)} | {sd} |"
        )
    worse = int((comparisons.mae_delta_model_minus_naive > 0).sum())
    contain_zero = int(((comparisons.bootstrap_ci_low <= 0)
                        & (comparisons.bootstrap_ci_high >= 0)).sum())
    ridge_cal = calibration[calibration.model == "ridge"]
    target = 0.9
    meet = int((ridge_cal.coverage_ready >= target).sum())
    lines.extend([
        "",
        "## Predeclared endpoints",
        f"- Naive beats Ridge on point-estimate MAE in **{worse}/{len(comparisons)}** comparisons "
        "(positive Ridge−Naive difference).",
        f"- **{contain_zero}/{len(comparisons)}** descriptive paired bootstrap intervals include zero; "
        "no across-market inferential conclusion.",
        f"- Ridge nominal-90% interval empirical coverage ≥90% in "
        f"**{meet}/{len(ridge_cal)}** asset/horizon cells; small sample/time dependence.",
        "- Selected MAE comparisons use only origins **both** models accepted; "
        "post-hoc conditioning is not proof that abstention improves model skill.",
        "",
        "## Data-integrity follow-up, not reanalysis",
        "- Review top absolute errors and raw-close corporate-action histories "
        "for any unusual jumps. **Do not exclude** observations from this run.",
        "- If a price issue is confirmed, document it as a limitation and specify "
        "a genuinely separate future validation; never relabel corrected results "
        "on these dates as the first independent evaluation.",
        "",
        "## Limitations",
        "- Historical dates overlap previously explored stock series; four new "
        "assets are not a prospective holdout.",
        "- No formal multi-asset adjusted hypothesis tests; block-bootstrap CIs "
        "are descriptive and use frozen block length/seed.",
        "- Vendor is Tiingo EOD raw Close, potentially impacted by corporate actions. "
        "No profitability, trading-system or causal claims.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path,
                        default=Path("reports/backtest/m5_tiingo_v2"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/backtest/m51_evidence_audit"))
    args = parser.parse_args(argv)
    spec = load_v2_protocol()
    predictions, original = verify_first_run(args.result_dir, spec)
    pairs, cal = recalculated_diagnostics(args.result_dir, spec)
    matched = pd.read_csv(args.result_dir / FILENAME_MAP["selected"])
    flags = descriptive_error_flags(predictions)
    report = generate_report(pairs, cal, matched, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        path.name: sha(path)
        for path in (args.result_dir / p for p in FILENAME_MAP.values())
    }
    manifest = {
        "protocol": PROTOCOL,
        "stage": "POST_HOC_AUDIT_OF_UNCHANGED_FIRST_EVALUATION",
        "frozen_protocol_sha256": LOCKED_V2_SHA256,
        "source_sha256": original["preflight_sources_sha256"],
        "input_artifact_sha256": input_hashes,
        "scores_recomputed_from_existing_predictions": True,
        "predictions_refitted": False,
        "outliers_excluded": False,
        "data_used_for_model_selection": False,
        "extra_post_hoc_diagnostic": "top 3 errors per asset/model/horizon; "
                                  "25% realized movement flag, not an exclusion rule",
    }
    (args.output_dir / "m51_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "m51_evidence_summary.md").write_text(report, encoding="utf-8")
    flags.to_csv(args.output_dir / "m51_top_errors_descriptive.csv", index=False)
    print("M5.1 PASS: original scores, paired bootstrap, interval coverage, "
          "same-origin selection and source hashes cross-checked")
    print(report)
    print(f"Evidence-only outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
