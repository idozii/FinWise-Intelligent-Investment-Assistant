"""Read-only R6 split-event audit of the original M5-v2 first evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.frozen_eval_tiingo import load_v2_protocol
from research.m5_evidence_audit import verify_first_run, recalculated_diagnostics

REGISTRY = Path(__file__).parent / "protocols" / "r6_verified_corporate_actions_v1.json"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_events(path=REGISTRY):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("registry_id") != "finwise-r6-official-splits-v1":
        raise ValueError("Incorrect official event registry")
    for e in registry["events"]:
        if e["event_type"] != "share_split" or float(e["ratio_new_per_old"]) <= 0:
            raise ValueError("Invalid share-split event")
        if not e["source_url"].startswith("https://"):
            raise ValueError("Missing official HTTPS evidence")
        if e["announcement_date"] > e["effective_date"]:
            raise ValueError("Announcement cannot follow effective date")
    return registry["events"]

def split_day_prices(csv: Path, event: dict) -> dict:
    """Independent price-scale observation, NOT adjusted model performance."""
    raw = pd.read_csv(csv, usecols=["Date", "Close"])
    dates = pd.to_datetime(raw.Date, utc=True, errors="raise").dt.normalize()
    prices = pd.to_numeric(raw.Close, errors="raise").to_numpy(float)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Unsorted or duplicate source dates")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid raw prices")
    index = np.flatnonzero(dates == pd.Timestamp(event["effective_date"], tz="UTC"))
    if len(index) != 1 or index[0] == 0:
        return {"split_bar_available": False, "note": "Exact split-date or prior bar absent"}
    j = int(index[0])
    return {
        "split_bar_available": True,
        "previous_trading_date": dates.iloc[j-1].date().isoformat(),
        "split_trading_date": dates.iloc[j].date().isoformat(),
        "previous_raw_close": float(prices[j-1]),
        "split_day_raw_close": float(prices[j]),
        "raw_price_ratio": float(prices[j] / prices[j-1]),
        "hypothetical_split_neutral_return": float(
            prices[j] * float(event["ratio_new_per_old"]) / prices[j-1] - 1
        ),
        "note": "One-session price-scale diagnostic; not a rescored evaluation",
    }

def annotate(predictions: pd.DataFrame, events: list[dict], sources: dict[str, Path]):
    rows = predictions.copy()
    origin = pd.to_datetime(rows.train_end, utc=True)
    target = pd.to_datetime(rows.target_date, utc=True)
    rows["abs_error"] = (rows.actual - rows.predicted).abs()
    rows["crosses_verified_split"] = False
    rows["post_split_origin_first_30_trading_days"] = False
    rows["event_announced_before_origin"] = False
    rows["event_id"] = ""
    checks = []
    for e in events:
        a = e["asset"]
        if a not in sources:
            continue
        split_date = pd.Timestamp(e["effective_date"], tz="UTC")
        announced = pd.Timestamp(e["announcement_date"], tz="UTC")
        price_dates = pd.to_datetime(
            pd.read_csv(sources[a], usecols=["Date"]).Date, utc=True
        ).dt.normalize()
        recent_dates = price_dates.loc[price_dates >= split_date].head(30)
        active = rows.asset == a
        cross = active & (origin < split_date) & (target >= split_date)
        recent = active & origin.dt.normalize().isin(recent_dates)
        rows.loc[cross, "crosses_verified_split"] = True
        rows.loc[recent, "post_split_origin_first_30_trading_days"] = True
        rows.loc[active & (origin >= announced), "event_announced_before_origin"] = True
        rows.loc[cross | recent, "event_id"] = a + ":" + e["effective_date"]
        checks.append({
            "asset": a, "event_effective_date": e["effective_date"],
            "event_announced": e["announcement_date"],
            "ratio_new_per_old": e["ratio_new_per_old"],
            "source_url": e["source_url"], **split_day_prices(sources[a], e),
        })
    if not checks:
        raise ValueError("No official events matched frozen evaluation assets")
    return rows, pd.DataFrame(checks)

def audit(result_dir: Path, output_dir: Path) -> dict:
    if result_dir.resolve() == output_dir.resolve():
        raise ValueError("R6 outputs cannot overwrite the frozen evaluation")
    spec = load_v2_protocol()
    pred, manifest = verify_first_run(result_dir, spec)
    recalculated_diagnostics(result_dir, spec)
    preflight = json.loads((result_dir/"preflight_manifest.json").read_text())
    sources = {s["asset"]: Path(s["file"]) for s in preflight["sources"]}
    rows, checks = annotate(pred, load_events(), sources)
    summaries = []
    for (asset, model, horizon), group in rows.groupby(
        ["asset", "model", "horizon_step"], sort=True
    ):
        crosses = group.loc[group.crosses_verified_split]
        summaries.append({
            "asset": asset, "model": model, "horizon_step": horizon,
            "n_origins": len(group),
            "n_crossing_split": len(crosses),
            "n_recent_post_split": int(group.post_split_origin_first_30_trading_days.sum()),
            "unchanged_original_mae": float(group.abs_error.mean()),
            "crossing_abs_error_sum": float(crosses.abs_error.sum()),
            "crossing_abs_error_fraction": float(
                crosses.abs_error.sum() / group.abs_error.sum()
            ) if group.abs_error.sum() else 0.,
            "max_abs_error": float(group.abs_error.max()),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.loc[
        rows.crosses_verified_split | rows.post_split_origin_first_30_trading_days,
        ["asset", "model", "horizon_step", "origin_index", "train_end",
         "target_date", "abs_error", "crosses_verified_split",
         "post_split_origin_first_30_trading_days", "event_announced_before_origin", "event_id"]
    ].to_csv(output_dir/"r6_event_linked_origins.csv", index=False)
    df = pd.DataFrame(summaries)
    df.to_csv(output_dir/"r6_event_impact_descriptive.csv", index=False)
    checks.to_csv(output_dir/"r6_split_date_raw_close_check.csv", index=False)
    file_names = [
        "preflight_manifest.json", "holdout_result_manifest.json",
        "holdout_predictions.csv", "holdout_metrics.csv",
        "holdout_paired_comparisons.csv", "holdout_calibration.csv",
        "holdout_matched_accepted.csv", "holdout_interval_predictions.csv",
    ]
    audit_manifest = {
        "protocol": "finwise-r6-corporate-action-audit-v1",
        "stage": "POST_HOC_EVENT_LINKED_DIAGNOSTICS",
        "original_protocol_sha256": manifest["protocol_sha256"],
        "original_source_sha256": manifest["preflight_sources_sha256"],
        "event_registry_sha256": digest(REGISTRY),
        "original_artifact_sha256": {name: digest(result_dir/name) for name in file_names},
        "refit": False, "excluded_origins": 0,
        "modified_original_scores": False,
        "limitations": "No recomputation of alternative strategy returns, no corporate-action-adjusted backtest, no new independent test",
    }
    (output_dir/"r6_audit_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2)+"\n", encoding="utf-8"
    )
    xle = df.loc[(df.asset == "XLE") & (df.horizon_step == 7)]
    lines = [
        "# R6 — Corporate action integrity audit",
        "",
        "Post-hoc investigation of the original M5-v2 data; frozen scores preserved.",
        "Official XLE 2-for-1 split effective 2025-12-05; announced 2025-11-20.",
        "Source: https://investors.statestreet.com/investor-news-events/press-releases/news-details/2025/State-Street-Investment-Management-Announces-Share-Splits-for-Five-Select-Sector-SPDR-ETFs/default.aspx",
        "",
        "| Model | Original MAE | Cross-split origins | Cross-split share of absolute error |",
        "|---|---:|---:|---:|",
    ]
    for row in xle.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.unchanged_original_mae:.3f} | "
            f"{row.n_crossing_split} | {row.crossing_abs_error_fraction:.1%} |"
        )
    lines += [
        "",
        "No original origins have been excluded or rescored. Split-neutral price "
        "movement is a standalone scale diagnostic only. Because the split "
        "was announced before the affected forecast, raw-close forecast "
        "comparisons across the split require particular caution.",
    ]
    (output_dir/"r6_integrity_report.md").write_text("\n".join(lines)+"\n")
    return audit_manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("reports/backtest/m5_tiingo_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/backtest/r6_corporate_actions"))
    args = parser.parse_args(argv)
    audit(args.result_dir, args.output_dir)
    print("R6 PASS: official split linked; first M5-v2 evaluation preserved")
    print(f"Outputs: {args.output_dir}")

if __name__ == "__main__":
    main()
