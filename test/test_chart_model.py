import chart_model

def test_fetch_price_history_basic():
    hist = chart_model.fetch_price_history("RELIANCE", chart_range="1y")
    assert isinstance(hist, list)

def test_run_analysis_structure():
    res = chart_model.run_analysis("RELIANCE", chart_range="1y")
    assert "forecast_price" in res
    assert "decision" in res
    assert "score" in res
