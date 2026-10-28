"""R11.1: research graphics must be numerical, complete, and conservative."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.r8_adaptive_development import compare
from research.r9_selective_risk import policy_rows
from research.r11_paper_evidence import summarize_r8, summarize_r9
from research.r111_scientific_figures import (
    PROTOCOL, generate, verified_m5_cells, verify_r8_score_rows,
)


def fixture_dataset(n=42):
    rows=[]
    dates=pd.date_range("2025-01-01",periods=n*8+20,tz="UTC")
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for model in ("naive","ridge"):
            for h in (1,7):
                for i in range(n):
                    k=i*7
                    last=100+.1*i
                    rows.append({
                        "asset":asset,"model":model,"horizon_step":h,
                        "origin_index":i+252,
                        "train_end":dates[k].date().isoformat(),
                        "target_date":dates[k+h].date().isoformat(),
                        "last_observed_close":last,
                        "actual":last+(20 if asset=="TSLA" and h==7 else 1.8)*np.sin(i/5)+.2,
                        "predicted":last+(0 if model=="naive" else .1),
                    })
    interval, scores=compare(pd.DataFrame(rows))
    curve,_=policy_rows(interval)
    return interval,scores,curve


def registry(scores,curve):
    cells=[]
    for asset in ("GLD","IWM","XLE","XLV"):
        for horizon in (1,7):
            cells.append({
                "asset":asset,"horizon":horizon,"n_origins":64,
                "naive_mae":2.0,"ridge_mae":2.1,
                "ridge_minus_naive_mae":.1,
                "paired_bootstrap_low":-.1,
                "paired_bootstrap_high":.2,
                "ridge_ready_coverage":.875,"ridge_n_accepted":25,
            })
    return {
        "protocol":"finwise-r11-manuscript-evidence-v1",
        "evidence_stage":"R11_DRAFT_MANUSCRIPT_HISTORICAL_AND_DEVELOPMENT_ONLY",
        "independent_r10_evaluation_performed":False,
        "m5_first_run":cells,
        "r8_development":summarize_r8(scores),
        "r9_development_ridge_h7_width_10pct":summarize_r9(curve),
    }


def test_complete_verified_figure_set_is_from_synthetic_dev_data(tmp_path):
    intervals, scores, curve=fixture_dataset()
    evidence=registry(scores,curve)
    out=generate(evidence,scores,intervals,curve,tmp_path/"figures")
    assert out["protocol"]==PROTOCOL
    assert out["r10_independent_evaluation_performed"] is False
    assert out["n_figures"]==7
    assert len(out["figure_sha256"])==14
    assert set(out["assets_m5"])=={"GLD","IWM","XLE","XLV"}
    assert set(out["assets_dev"])=={"AAPL","MSFT","SPY","TSLA"}
    for name,digest in out["figure_sha256"].items():
        assert Path(tmp_path/"figures"/name).is_file()
        assert len(digest)==64
    assert (tmp_path/"figures"/"fig03_r9_tsla_risk_availability.svg").exists()


def test_refuse_forged_r10_independent_claims():
    _,scores,curve=fixture_dataset()
    evidence=registry(scores,curve)
    evidence["independent_r10_evaluation_performed"]=True
    with pytest.raises(ValueError,match="R10-blocked"):
        verified_m5_cells(evidence)


def test_refuse_missing_m5_comparison_or_delta_tampering():
    _,scores,curve=fixture_dataset()
    evidence=registry(scores,curve)
    evidence["m5_first_run"].pop()
    with pytest.raises(ValueError,match="eight M5"):
        verified_m5_cells(evidence)
    evidence=registry(scores,curve)
    evidence["m5_first_run"][0]["ridge_minus_naive_mae"]=.5
    with pytest.raises(ValueError,match="arithmetic mismatch"):
        verified_m5_cells(evidence)


def test_r8_reaggregates_and_rejects_tampered_coverage():
    interval,scores,curve=fixture_dataset()
    evidence=registry(scores,curve)
    assert len(verify_r8_score_rows(interval,scores,evidence))==32
    corrupted=scores.copy()
    corrupted.loc[0,"coverage_ready"]=.999
    with pytest.raises(ValueError,match="coverage_ready"):
        verify_r8_score_rows(interval,corrupted,evidence)


def test_r9_rejects_tampered_selected_risk_before_paper_figures(tmp_path):
    interval,scores,curve=fixture_dataset()
    evidence=registry(scores,curve)
    corrupted=curve.copy()
    corrupted.loc[0,"relative_mae_selected"]=0.0
    with pytest.raises(ValueError,match="Saved R9 relative_mae_selected"):
        generate(evidence,scores,interval,corrupted,tmp_path/"forged")


def test_missing_accepted_forecasts_are_not_zero_risk(tmp_path):
    interval,scores,curve=fixture_dataset()
    evidence=registry(scores,curve)
    # A valid zero-accepted policy case must have NA, not a falsely perfect 0.
    zeros=curve.loc[curve.n_accepted==0]
    assert len(zeros)>0
    assert zeros.relative_mae_selected.isna().all()
    output=generate(evidence,scores,interval,curve,tmp_path/"zero")
    assert "No accepted forecasts means unavailable selected loss, not zero risk" in (
        output["interpretation"]
    )
