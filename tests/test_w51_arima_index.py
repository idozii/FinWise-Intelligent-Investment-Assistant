"""W5.1: ARIMA must not rely on a gapped unsupported statsmodels index."""
import warnings

import numpy as np
import pandas as pd

import backend.app as application


def test_arima_ignores_gapped_pandas_row_ids_not_price_order():
    values = 100 + .03 * np.arange(90) + 1.2 * np.sin(np.arange(90)/7)
    contiguous = pd.Series(values)
    irregular = pd.Series(values, index=[(i+20)*3 for i in range(len(values))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p1, lo1, hi1 = application._forecast_arima(irregular, days=7)
        p2, lo2, hi2 = application._forecast_arima(contiguous, days=7)
    index_warnings = [
        str(w.message) for w in caught
        if "unsupported index" in str(w.message).lower()
        or "No supported index" in str(w.message)
    ]
    assert not index_warnings
    assert len(p1) == len(lo1) == len(hi1) == 7
    assert np.isfinite(np.r_[p1, lo1, hi1]).all()
    np.testing.assert_allclose(p1, p2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(lo1, lo2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(hi1, hi2, rtol=1e-8, atol=1e-8)


def test_arima_empty_or_short_history_still_uses_previous_fallback():
    p, lower, upper = application._forecast_arima(pd.Series(np.linspace(100, 110, 20)), 3)
    assert len(p) == len(lower) == len(upper) == 3
