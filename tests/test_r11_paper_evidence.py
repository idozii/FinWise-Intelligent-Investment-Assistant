"""R11 paper claims must be derived from typed evidence, never invented."""
import numpy as np
import pandas as pd
import pytest

from research.r11_paper_evidence import (
    PROTOCOL, render_tables, summarize_m5, summarize_r8, summarize_r9,
)


def synthetic_first_run_tables():
    metrics=[]
    paired=[]
    calibration=[]
    matched=[]
    for asset in ("GLD","IWM","XLE","XLV"):
        for h in (1,7):
            for model,mae in (("naive",2.),("ridge",2.1)):
                metrics.append({"asset":asset,"model":model,"horizon_step":h,
                                "n_origins":64,"mae":mae})
            paired.append({"asset":asset,"model":"ridge","horizon_step":h,
                           "mae_delta_model_minus_naive":.1,
                           "bootstrap_ci_low":-.1,"bootstrap_ci_high":.2})
            calibration.append({"asset":asset,"model":"ridge","horizon_step":h,
                                "n_ready":64,"n_accepted":32,
                                "coverage_ready":.875})
            matched.append({"asset":asset,"model":"ridge","horizon_step":h,
                            "n_both_accepted":20,
                            "delta_mae_model_minus_naive_same_accepted":-.05})
    return tuple(pd.DataFrame(x) for x in (metrics,paired,calibration,matched))


def synthetic_r8():
    rows=[]
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for model in ("naive","ridge"):
            for h in (1,7):
                for method,coverage,width in (("fixed_rolling",.83,.06),
                                               ("delayed_feedback_aci_style",.89,.07)):
                    rows.append({"asset":asset,"model":model,"horizon_step":h,
                                 "calibration_method":method,
                                 "coverage_ready":coverage,
                                 "mean_relative_width_ready":width})
    return pd.DataFrame(rows)


def synthetic_r9():
    rows=[]
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for method in ("fixed_rolling","delayed_feedback_aci_style"):
            rows.append({"asset":asset,"model":"ridge","horizon_step":7,
                         "calibration_method":method,"max_relative_width":.10,
                         "n_origins":143,"n_accepted":0 if asset=="TSLA" else 20,
                         "interval_coverage_selected":np.nan if asset=="TSLA" else .8,
                         "relative_mae_selected":np.nan if asset=="TSLA" else .02})
    return pd.DataFrame(rows)


def test_first_run_table_has_all_eight_groups_and_valid_deltas():
    rows=summarize_m5(*synthetic_first_run_tables())
    assert len(rows)==8
    assert all(r["ridge_minus_naive_mae"]>.0 for r in rows)
    assert all(r["n_origins"]==64 for r in rows)


def test_tampered_m5_delta_is_rejected():
    tables=list(synthetic_first_run_tables())
    tables[1].loc[0,"mae_delta_model_minus_naive"]=.7
    with pytest.raises(ValueError,match="MAE arithmetic"):
        summarize_m5(*tables)


def test_r8_coverage_and_width_counts_no_invented_performance():
    actual=summarize_r8(synthetic_r8())
    assert actual["n_comparison_groups"]==16
    assert actual["coverage_increased_groups"]==16
    assert actual["interval_width_increased_groups"]==16


def test_r9_zero_selected_origin_risk_is_missing():
    actual=summarize_r9(synthetic_r9())
    assert len(actual)==8
    zero=[x for x in actual if x["asset"]=="TSLA"]
    assert all(r["selected_relative_mae"] is None for r in zero)
    payload={
        "m5_first_run":summarize_m5(*synthetic_first_run_tables()),
        "r8_development":summarize_r8(synthetic_r8()),
        "r9_development_ridge_h7_width_10pct":actual,
    }
    text=render_tables(payload)
    assert "N/A" in text
    assert "NOT an independent validation" in text


def test_manuscript_is_honest_about_corporate_action_and_r10():
    from pathlib import Path
    doc=(Path(__file__).resolve().parents[1]/
         "research/paper/R11_MANUSCRIPT_DRAFT.md").read_text()
    assert "No new independent R10 evaluation has been executed" in doc
    assert "24.9%" in doc and "22.5%" in doc
    assert "No novel algorithm claim" in doc
    assert "Gibbs and Candès" in doc and "Sale and Ramdas" in doc
    assert "Stat" not in PROTOCOL or PROTOCOL.startswith("finwise-r11")
