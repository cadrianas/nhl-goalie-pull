"""Paths and data-source configuration.

Data source: MoneyPuck shot-level data. The official moneypuck.com download
paths have moved around; the peter-tanner.com mirror currently serves the same
per-season shots zips reliably (verified application/zip). Override BASE_URL via
the GOALIE_PULL_SHOTS_BASE_URL env var if the mirror changes or you host locally.

Each season file is `shots_<season>.zip` containing `shots_<season>.csv`, where
`<season>` is the year the season STARTED (2023 == 2023-24 season).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parents[1]  # src/goalie_pull -> project root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Data source -----------------------------------------------------------
BASE_URL = os.environ.get(
    "GOALIE_PULL_SHOTS_BASE_URL",
    "https://peter-tanner.com/moneypuck/downloads",
)

# Seasons to use by default (season == starting year). 2020 excluded by default
# as a COVID-shortened / no-fans outlier; add it back explicitly if wanted.
DEFAULT_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]

REGULATION_SECONDS = 3 * 1200  # 3 periods x 20 min


def shots_zip_url(season: int) -> str:
    return f"{BASE_URL}/shots_{season}.zip"


def raw_zip_path(season: int) -> Path:
    return RAW_DIR / f"shots_{season}.zip"


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / name
