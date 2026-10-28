"""R11.2 visual summaries: no threshold collision or imaginary no-risk points."""
import numpy as np
import pandas as pd
import pytest

from research.r111_scientific_figures import (
    PROTOCOL, r9_policy_table, plot_r9
)


def synthetic_policy(asset="SPY", zero_widths=()):
    rows=[]
    for method in ("fixed_rolling", "delayed_feedback_aci_style"):
        for threshold in (.05, .10, .15, .20, .30):
            accepted=0 if threshold in zero_widths else (
                30 if threshold <= .10 else 45
            )
            # Consecutive higher widths can map to exactly the same
            # accepted origins. Their policy rows MUST remain visible.
            rows.append({
                "asset":asset, "model":"ridge", "horizon_step":7,
                "calibration_method":method,
                "max_relative_width":threshold,
                "n_origins":143, "n_accepted":accepted,
                "acceptance_rate_all":accepted/143,
                "relative_mae_selected":np.nan if accepted==0 else .018,
            })
    return pd.DataFrame(rows)


def test_all_width_thresholds_visible_even_when_plot_points_repeat():
    data=synthetic_policy()
    rows=r9_policy_table(data)
    assert len(rows)==5
    assert [r[0] for r in rows]==["5%","10%","15%","20%","30%"]
    assert rows[2][1:] == rows[3][1:] == rows[4][1:]
    assert all(len(row)==5 for row in rows)


def test_zero_accepted_is_na_not_zero_error():
    data=synthetic_policy(zero_widths=(.05,.10))
    rows=r9_policy_table(data)
    assert rows[0]==["5%","0/143","N/A","0/143","N/A"]
    assert rows[1]==["10%","0/143","N/A","0/143","N/A"]


def test_selected_n_below_30_flagged_in_exact_table():
    data=synthetic_policy()
    mask=data.max_relative_width == .10
    data.loc[mask,"n_accepted"]=17
    data.loc[mask,"acceptance_rate_all"]=17/143
    rows=r9_policy_table(data)
    assert rows[1][1]=="17/143"
    assert rows[1][2]=="1.80% *"
    assert rows[1][4]=="1.80% *"


def test_inconsistent_zero_risk_rejected():
    data=synthetic_policy(zero_widths=(.05,))
    data.loc[data.max_relative_width==.05,"relative_mae_selected"]=0.0
    with pytest.raises(ValueError,match="missing selected risk"):
        r9_policy_table(data)


def test_missing_or_duplicate_threshold_rejected():
    data=synthetic_policy()
    with pytest.raises(ValueError,match="exactly two methods"):
        r9_policy_table(data.iloc[:-1])
    dup=data.copy()
    dup.loc[dup.index[0],"max_relative_width"]=.10
    with pytest.raises(ValueError,match="exactly two methods"):
        r9_policy_table(dup)


def test_plot_does_not_change_source_and_writes_versioned_formats(tmp_path):
    input_data=synthetic_policy(zero_widths=(.05,.10))
    baseline=input_data.copy(deep=True)
    produced=plot_r9(input_data,"SPY",tmp_path)
    assert len(produced)==2
    assert (tmp_path/"fig03_r9_spy_risk_availability.png").is_file()
    assert (tmp_path/"fig03_r9_spy_risk_availability.svg").is_file()
    pd.testing.assert_frame_equal(input_data,baseline)
    assert "r112" in PROTOCOL
