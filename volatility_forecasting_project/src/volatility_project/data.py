from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FF5_FIZ_ARCHIVE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/ftp_202412/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: Path, refresh: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size and not refresh:
        return path
    headers = {"User-Agent": "volatility-forecasting-research/1.0 (educational reproducibility)"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"Empty response from {url}")
    path.write_bytes(response.content)
    return path


def download_sources(raw_dir: str | Path = "data/raw", refresh: bool = False) -> dict:
    raw_dir = Path(raw_dir)
    ff_ciz_path = _download(FF5_URL, raw_dir / "fama_french_5_factors_daily_ciz.zip", refresh)
    ff_fiz_path = _download(FF5_FIZ_ARCHIVE_URL, raw_dir / "fama_french_5_factors_daily_fiz_2024.zip", refresh)
    vix_path = _download(FRED_VIX_URL, raw_dir / "fred_vixcls.csv", refresh)
    return {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "fama_french_5_factors_daily_current_ciz": {"url": FF5_URL, "file": str(ff_ciz_path), "sha256": sha256_file(ff_ciz_path), "bytes": ff_ciz_path.stat().st_size},
            "fama_french_5_factors_daily_dec_2024_fiz_archive": {"url": FF5_FIZ_ARCHIVE_URL, "file": str(ff_fiz_path), "sha256": sha256_file(ff_fiz_path), "bytes": ff_fiz_path.stat().st_size},
            "fred_vixcls": {"url": FRED_VIX_URL, "file": str(vix_path), "sha256": sha256_file(vix_path), "bytes": vix_path.stat().st_size},
        },
    }


def parse_fama_french_zip(zip_path: str | Path) -> pd.DataFrame:
    """Parse the daily section of an official Fama/French ZIP or extracted CSV."""
    path = Path(zip_path)
    payload = path.read_bytes()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
            if not names:
                raise ValueError("No CSV/TXT file found in Fama/French ZIP")
            raw = zf.read(names[0]).decode("utf-8-sig", errors="replace")
    else:
        # The publisher's extracted CSV is also a valid input; it begins with
        # metadata rows before the daily table header.
        raw = payload.decode("utf-8-sig", errors="replace")
    lines = raw.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if "Mkt-RF" in line and "SMB" in line and "RF" in line), None)
    if header_idx is None:
        raise ValueError("Could not locate the Fama/French daily table header")
    table_lines = []
    for line in lines[header_idx:]:
        if not line.strip():
            if table_lines:
                break
            continue
        first = line.split(",", 1)[0].strip().strip('"')
        if table_lines and not first.isdigit():
            break
        table_lines.append(line)
    df = pd.read_csv(io.StringIO("\n".join(table_lines)), skipinitialspace=True)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df = df[df["date"].astype(str).str.fullmatch(r"\d{8}")].copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected Fama/French column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "Mkt-RF", "RF"]).sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates in Fama/French data")
    return df.set_index("date")


def parse_fred_vix(csv_path: str | Path) -> pd.DataFrame:
    # FRED exports may be comma- or semicolon-delimited depending on the
    # download route/locale. Sniff the delimiter instead of assuming commas.
    df = pd.read_csv(csv_path, sep=None, engine="python")
    date_col = next((c for c in df.columns if str(c).strip().lower() in {"observation_date", "date"}), df.columns[0])
    value_col = next((c for c in df.columns if str(c).strip().upper() == "VIXCLS"), None)
    if value_col is None:
        raise ValueError("Could not locate VIXCLS column in FRED CSV")
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "vix"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    out = out.dropna(subset=["date", "vix"]).sort_values("date")
    out = out[out["vix"] > 0]
    if out["date"].duplicated().any():
        raise ValueError("Duplicate dates in VIX data")
    return out.set_index("date")


def build_market_panel(ff_zip: str | Path, vix_csv: str | Path, start_date: str = "1990-01-01") -> pd.DataFrame:
    ff = parse_fama_french_zip(ff_zip)
    vix = parse_fred_vix(vix_csv)
    # Preserve every market trading session. VIX is a predictor, so a missing
    # VIX observation should only exclude that row from VIX-dependent models;
    # it must never remove a market return and shift forward realised targets.
    panel = ff.join(vix, how="left").sort_index()
    panel = panel.loc[pd.Timestamp(start_date):].copy()
    # French factors and RF are reported in percent. Convert to decimal returns.
    for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        panel[col.lower().replace("-", "_")] = panel[col] / 100.0
    panel["market_excess"] = panel["mkt_rf"]
    panel["market_return"] = panel["mkt_rf"] + panel["rf"]
    panel["vix_decimal"] = panel["vix"] / 100.0
    panel["vix_implied_variance"] = panel["vix_decimal"] ** 2
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["market_return", "market_excess", "rf"])
    if panel.empty:
        raise ValueError("No overlapping observations after merging market and VIX data")
    if panel.index.duplicated().any():
        raise ValueError("Duplicate dates in merged market panel")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("Merged panel dates are not sorted")
    return panel


def compare_data_vintages(ciz_panel: pd.DataFrame, fiz_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarise differences between current CIZ-based and Dec-2024 FIZ archive panels."""
    common = ciz_panel.index.intersection(fiz_panel.index)
    if len(common) == 0:
        raise ValueError("The current and FIZ archive panels have no common dates")
    variables = [c for c in ("market_return", "market_excess", "rf", "smb", "hml", "rmw", "cma")
                 if c in ciz_panel.columns and c in fiz_panel.columns]
    rows = []
    for variable in variables:
        current = ciz_panel.loc[common, variable].astype(float)
        archive = fiz_panel.loc[common, variable].astype(float)
        difference = current - archive
        correlation = float(current.corr(archive)) if current.std() > 0 and archive.std() > 0 else np.nan
        rows.append({
            "variable": variable,
            "n_common": int(len(common)),
            "overlap_start": str(common.min().date()),
            "overlap_end": str(common.max().date()),
            "correlation_ciz_vs_fiz": correlation,
            "mean_absolute_difference": float(difference.abs().mean()),
            "max_absolute_difference": float(difference.abs().max()),
            "mean_difference_ciz_minus_fiz": float(difference.mean()),
        })
    return pd.DataFrame(rows)


def save_metadata(metadata: dict, path: str | Path = "data/processed/source_metadata.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
