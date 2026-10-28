"""R11 immutable evidence package for a scientifically qualified manuscript.

Regenerates paper tables from the already-audited M5 first run and
development-only R8/R9 artifacts. Never retrains or downloads market data.
Not a new R10 independent test; outputs are meant for evidence review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.frozen_eval_tiingo import load_v2_protocol
from research.m5_evidence_audit import verify_first_run, recalculated_diagnostics
from research.r9_selective_risk import load_dev_input, policy_rows

PROTOCOL="finwise-r11-manuscript-evidence-v1"
R8_METHODS=("fixed_rolling","delayed_feedback_aci_style")


def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_m5(metrics:pd.DataFrame,pairs:pd.DataFrame,
                 calibration:pd.DataFrame,matched:pd.DataFrame)->list[dict]:
    rows=[]
    for asset in ("GLD","IWM","XLE","XLV"):
        for h in (1,7):
            def only(d,model=None):
                q=(d.asset==asset)&(d.horizon_step==h)
                if model is not None:q &= d.model==model
                subset=d.loc[q]
                if len(subset)!=1:raise ValueError("M5 group missing or duplicated")
                return subset.iloc[0]
            naive=only(metrics,"naive")
            ridge=only(metrics,"ridge")
            paired=only(pairs,"ridge")
            cal=only(calibration,"ridge")
            joint=only(matched,"ridge")
            if (int(naive.n_origins)!=64 or int(ridge.n_origins)!=64
                or int(cal.n_ready)!=64):
                raise ValueError("M5 original first-run must have 64 origins per cell")
            delta=float(ridge.mae)-float(naive.mae)
            if not np.isclose(delta,paired.mae_delta_model_minus_naive,atol=1e-8):
                raise ValueError("M5 Ridge-Naive MAE arithmetic mismatch")
            rows.append({
                "asset":asset,"horizon":h,"n_origins":64,
                "naive_mae":float(naive.mae),"ridge_mae":float(ridge.mae),
                "ridge_minus_naive_mae":delta,
                "paired_bootstrap_low":float(paired.bootstrap_ci_low),
                "paired_bootstrap_high":float(paired.bootstrap_ci_high),
                "ridge_ready_coverage":float(cal.coverage_ready),
                "ridge_n_accepted":int(cal.n_accepted),
                "n_both_accepted":int(joint.n_both_accepted),
                "same_origin_selected_mae_delta":(
                    None if pd.isna(joint.delta_mae_model_minus_naive_same_accepted)
                    else float(joint.delta_mae_model_minus_naive_same_accepted)
                ),
            })
    return rows


def summarize_r8(scores:pd.DataFrame)->dict:
    key=["asset","model","horizon_step"]
    if len(scores)!=32 or scores.duplicated(key+["calibration_method"]).any():
        raise ValueError("R8 must contain exactly 32 unique dev method groups")
    if (set(scores.calibration_method)!=set(R8_METHODS)
        or set(scores.asset)!={"AAPL","MSFT","SPY","TSLA"}
        or set(scores.model)!={"naive","ridge"} or set(scores.horizon_step)!={1,7}):
        raise ValueError("Unexpected R8 dev model group")
    fixed=scores[scores.calibration_method=="fixed_rolling"].set_index(key).sort_index()
    adapt=scores[scores.calibration_method=="delayed_feedback_aci_style"].set_index(key).sort_index()
    if not fixed.index.equals(adapt.index):
        raise ValueError("R8 dev methods use different comparison groups")
    w=(adapt.mean_relative_width_ready-fixed.mean_relative_width_ready)
    cover=(adapt.coverage_ready-fixed.coverage_ready)
    if not np.isfinite(w).all() or not np.isfinite(cover).all():
        raise ValueError("R8 summary has unready/invalid coverage")
    return {
        "n_comparison_groups":16,
        "coverage_increased_groups":int((cover>1e-10).sum()),
        "coverage_unchanged_groups":int((np.abs(cover)<=1e-10).sum()),
        "coverage_decreased_groups":int((cover<-1e-10).sum()),
        "interval_width_increased_groups":int((w>1e-10).sum()),
        "interval_width_decreased_groups":int((w<-1e-10).sum()),
        "method_label":"ACI-style delayed feedback, exploratory only",
    }


def summarize_r9(curve:pd.DataFrame)->list[dict]:
    q=curve[(curve.model=="ridge")&(curve.horizon_step==7)&
            (np.isclose(curve.max_relative_width,.1))]
    if len(q)!=8 or q.duplicated(["asset","calibration_method"]).any():
        raise ValueError("R9 10% Ridge H7 requires 8 unique dev cells")
    rows=[]
    for r in q.sort_values(["asset","calibration_method"]).itertuples(index=False):
        rows.append({
            "asset":r.asset,
            "method":r.calibration_method,
            "n_origins":int(r.n_origins),
            "n_accepted":int(r.n_accepted),
            "selected_coverage":None if pd.isna(r.interval_coverage_selected)
                                else float(r.interval_coverage_selected),
            "selected_relative_mae":None if pd.isna(r.relative_mae_selected)
                                    else float(r.relative_mae_selected),
        })
    return rows


def render_tables(payload:dict)->str:
    lines=[
        "# R11 Manuscript evidence tables (generated from original artifacts)",
        "",
        "**Historical first run (M5-v2)** — descriptive retrospective external-asset "
        "replication, not prospective validation.",
        "",
        "| Asset | H | Naive MAE | Ridge MAE | Δ Ridge−Naive | Bootstrap 95% CI | Ridge coverage (ready) | Ridge accepted |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in payload["m5_first_run"]:
        lines.append(
            f"| {r['asset']} | {r['horizon']} | {r['naive_mae']:.3f} | "
            f"{r['ridge_mae']:.3f} | {r['ridge_minus_naive_mae']:+.3f} | "
            f"[{r['paired_bootstrap_low']:+.3f}, {r['paired_bootstrap_high']:+.3f}] | "
            f"{100*r['ridge_ready_coverage']:.1f}% | {r['ridge_n_accepted']}/64 |"
        )
    summ=payload["r8_development"]
    lines+=["","**R8 development-only** — coverage rose in "
             f"{summ['coverage_increased_groups']}/16 groups; ready-origin "
             f"interval width rose in {summ['interval_width_increased_groups']}/16. "
             "These descriptive comparisons are NOT an independent validation.",
             "","**R9 development-only, Ridge H7, 10% width**","",
             "| Asset | Method | Accepted | Selected coverage | Selected relative MAE |",
             "|---|---|---:|---:|---:|"]
    for r in payload["r9_development_ridge_h7_width_10pct"]:
        coverage="N/A" if r["selected_coverage"] is None else f"{100*r['selected_coverage']:.1f}%"
        mae="N/A" if r["selected_relative_mae"] is None else f"{100*r['selected_relative_mae']:.2f}%"
        lines.append(f"| {r['asset']} | {r['method']} | {r['n_accepted']}/{r['n_origins']} | {coverage} | {mae} |")
    lines+=["","All selected-risk comparisons condition on potentially different "
             "origin subsets. Zero selected forecasts means unavailable estimate, "
             "never zero error. Corporate actions may affect raw Close MAE; "
             "M5 original numbers are preserved.",
             ""]
    return "\n".join(lines)


def compile_evidence(m5:Path,r8:Path,r9:Path,dev_source:Path,
                     r6_manifest:Path)->dict:
    spec=load_v2_protocol()
    _,original=verify_first_run(m5,spec)
    recalculated_diagnostics(m5,spec)
    r6=json.loads(r6_manifest.read_text())
    if (r6.get("protocol")!="finwise-r6-corporate-action-audit-v1"
        or r6.get("refit") is not False or r6.get("excluded_origins")!=0
        or r6.get("modified_original_scores") is not False):
        raise ValueError("R6 original-evidence audit failed")
    for filename,expected in r6["original_artifact_sha256"].items():
        if digest(m5/filename)!=expected:
            raise ValueError("First-run input changed since R6")
    frame=load_dev_input(r8/"r8_intervals_development.csv",
                         r8/"r8_manifest.json",dev_source)
    recomputed,_=policy_rows(frame)
    prior=pd.read_csv(r9/"r9_risk_coverage_dev.csv")
    keys=["asset","model","horizon_step","calibration_method","max_relative_width"]
    merged=prior.merge(recomputed,on=keys,how="outer",indicator=True,
                        suffixes=("_saved","_check"),validate="one_to_one")
    if len(merged)!=160 or (merged._merge!="both").any():
        raise ValueError("R9 saved risk/availability grid incomplete")
    for col in ["n_accepted","acceptance_rate_all","interval_coverage_selected",
                "relative_mae_selected","interval_coverage_ready"]:
        if not np.allclose(merged[col+"_saved"],merged[col+"_check"],
                           equal_nan=True,atol=1e-9,rtol=1e-8):
            raise ValueError("R9 saved policy arithmetic changed")
    m5_cells=summarize_m5(
        pd.read_csv(m5/"holdout_metrics.csv"),
        pd.read_csv(m5/"holdout_paired_comparisons.csv"),
        pd.read_csv(m5/"holdout_calibration.csv"),
        pd.read_csv(m5/"holdout_matched_accepted.csv"),
    )
    return {
        "protocol":PROTOCOL,
        "evidence_stage":"R11_DRAFT_MANUSCRIPT_HISTORICAL_AND_DEVELOPMENT_ONLY",
        "independent_r10_evaluation_performed":False,
        "m5_frozen_protocol_sha256":original["protocol_sha256"],
        "r6_manifest_sha256":digest(r6_manifest),
        "r8_manifest_sha256":digest(r8/"r8_manifest.json"),
        "r9_saved_policy_csv_sha256":digest(r9/"r9_risk_coverage_dev.csv"),
        "m5_first_run":m5_cells,
        "r8_development":summarize_r8(pd.read_csv(r8/"r8_comparison_development.csv")),
        "r9_development_ridge_h7_width_10pct":summarize_r9(recomputed),
        "warnings":[
            "No independent R10 evaluation has occurred",
            "No universal forecast skill or theoretical coverage guarantee",
            "No profitability or causal effect claimed",
            "M5 raw Close XLE split-crossing origin remains in original scores",
            "R8/R9 development policies were inspected post hoc",
        ]
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m5-dir",type=Path,default=Path("reports/backtest/m5_tiingo_v2"))
    p.add_argument("--r6-manifest",type=Path,default=Path("reports/backtest/r6_corporate_actions/r6_audit_manifest.json"))
    p.add_argument("--r8-dir",type=Path,default=Path("reports/backtest/r8_development"))
    p.add_argument("--r9-dir",type=Path,default=Path("reports/backtest/r9_risk_coverage_dev"))
    p.add_argument("--dev-predictions",type=Path,default=Path("reports/backtest/m21_multiasset/asset_predictions.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("reports/backtest/r11_manuscript_evidence"))
    args=p.parse_args(argv)
    if any(args.output_dir.resolve().is_relative_to(d.resolve()) for d in
           (args.m5_dir,args.r8_dir,args.r9_dir)):
        raise ValueError("Manuscript outputs cannot overwrite evidence inputs")
    result=compile_evidence(args.m5_dir,args.r8_dir,args.r9_dir,
                            args.dev_predictions,args.r6_manifest)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"r11_evidence_registry.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output_dir/"r11_generated_tables.md").write_text(render_tables(result))
    print("R11 EVIDENCE VERIFIED: M5 + R6 + R8 + R9; no independent R10 test")
    print(f"Tables: {args.output_dir/'r11_generated_tables.md'}")


if __name__=="__main__":
    main()
