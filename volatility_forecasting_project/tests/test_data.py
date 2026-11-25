import sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import numpy as np
from volatility_project.data import parse_fama_french_zip, parse_fred_vix, compare_data_vintages


def test_parse_small_french_zip(tmp_path):
    content = """This file was created for testing.\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20200102,0.10,0.01,0.02,0.03,0.04,0.00\n20200103,-0.20,0.02,0.01,0.00,-0.01,0.00\n\nAnnual Factors: January-December\n"""
    z = tmp_path / "ff.zip"
    with zipfile.ZipFile(z, "w") as f: f.writestr("sample.csv", content)
    got = parse_fama_french_zip(z)
    assert len(got) == 2
    assert got.index[0] == pd.Timestamp("2020-01-02")
    assert got.iloc[0]["Mkt-RF"] == 0.10


def test_parse_fred_vix(tmp_path):
    p = tmp_path / "vix.csv"
    p.write_text("observation_date,VIXCLS\n2020-01-02,12.47\n2020-01-03,.\n2020-01-06,13.85\n")
    got = parse_fred_vix(p)
    assert len(got) == 2
    assert got.iloc[0]["vix"] == 12.47


def test_compare_data_vintages_reports_common_history():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    current = pd.DataFrame({"market_return": [0.01, 0.02, -0.01],
                            "market_excess": [0.009, 0.019, -0.011],
                            "rf": [0.001, 0.001, 0.001]}, index=idx)
    archive = pd.DataFrame({"market_return": [0.01, 0.021, -0.01],
                            "market_excess": [0.009, 0.020, -0.011],
                            "rf": [0.001, 0.001, 0.001]}, index=idx)
    result = compare_data_vintages(current, archive)
    assert set(result["variable"]) == {"market_return", "market_excess", "rf"}
    assert (result["n_common"] == 3).all()
    market_row = result.set_index("variable").loc["market_return"]
    assert np.isclose(market_row["max_absolute_difference"], 0.001)


def test_parse_fama_french_extracted_csv(tmp_path):
    content = "This file was created for testing.\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20200102,0.10,0.01,0.02,0.03,0.04,0.00\n20200103,-0.20,0.02,0.01,0.00,-0.01,0.00\n"
    p = tmp_path / "ff.csv"
    p.write_text(content)
    got = parse_fama_french_zip(p)
    assert len(got) == 2
    assert got.index[-1] == pd.Timestamp("2020-01-03")


def test_parse_fred_vix_semicolon_delimiter(tmp_path):
    p = tmp_path / "vix_semicolon.csv"
    p.write_text("observation_date;VIXCLS\n2020-01-02;12.47\n2020-01-03;.\n2020-01-06;13.85\n")
    got = parse_fred_vix(p)
    assert len(got) == 2
    assert got.iloc[1]["vix"] == 13.85
