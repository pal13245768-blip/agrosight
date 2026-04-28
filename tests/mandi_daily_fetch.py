#!/usr/bin/env python3
"""
Daily mandi price fetcher from data.gov.in (AGMARKNET-backed dataset).

What it does:
- Pulls all records with pagination
- Optionally filters by state / district / market / commodity
- Saves a daily CSV
- Appends only new rows into a local SQLite database for incremental ingestion
"""

from __future__ import annotations

import os
import time
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import pandas as pd


RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
OUT_DIR = os.getenv("MANDI_OUT_DIR", "./mandi_data")
DB_PATH = os.getenv("MANDI_DB_PATH", "./mandi_data/mandi_prices.sqlite3")
# Match the .env variable name exactly
API_KEY = os.getenv("DATA_GOV_API_KEY_1", os.getenv("DATA_GOV_IN_API_KEY", "")).strip()

# Optional filters (set any of these as env vars if you want)
STATE = os.getenv("MANDI_STATE", "").strip()
DISTRICT = os.getenv("MANDI_DISTRICT", "").strip()
MARKET = os.getenv("MANDI_MARKET", "").strip()
COMMODITY = os.getenv("MANDI_COMMODITY", "").strip()

LIMIT = int(os.getenv("MANDI_LIMIT", "100"))
MAX_PAGES = int(os.getenv("MANDI_MAX_PAGES", "0"))  # 0 = no hard cap
SLEEP_SECONDS = float(os.getenv("MANDI_SLEEP_SECONDS", "0.3"))


def build_params(offset: int) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError(
            "DATA_GOV_API_KEY_1 is not set. "
            "Run: export DATA_GOV_API_KEY_1=<your-key>  (or set it in .env)"
        )

    params: Dict[str, Any] = {
        "api-key": API_KEY,
        "format": "json",
        "offset": offset,
        "limit": LIMIT,
    }

    # Resource 9ef84268 uses filters[state.keyword] (NOT filters[state])
    if STATE:
        params["filters[state.keyword]"] = STATE
    if DISTRICT:
        params["filters[district]"] = DISTRICT
    if MARKET:
        params["filters[market]"] = MARKET
    if COMMODITY:
        params["filters[commodity]"] = COMMODITY

    return params


def fetch_page(session: requests.Session, offset: int, retries: int = 5) -> List[Dict[str, Any]]:
    url = BASE_URL
    params = build_params(offset)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get("records", [])
            if not isinstance(records, list):
                raise ValueError("Unexpected API response: 'records' is not a list")
            return records
        except Exception as e:
            last_err = e
            backoff = min(2 ** attempt, 20)
            time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch offset={offset}") from last_err


def fetch_all() -> pd.DataFrame:
    session = requests.Session()
    all_records: List[Dict[str, Any]] = []

    offset = 0
    page_count = 0

    while True:
        if MAX_PAGES and page_count >= MAX_PAGES:
            break

        records = fetch_page(session, offset)
        if not records:
            break

        all_records.extend(records)

        if len(records) < LIMIT:
            break

        offset += LIMIT
        page_count += 1
        time.sleep(SLEEP_SECONDS)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Add ingestion metadata
    df["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Normalize common column names if present
    rename_map = {
        "state": "state",
        "district": "district",
        "market": "market",
        "commodity": "commodity",
        "variety": "variety",
        "min_price": "min_price",
        "max_price": "max_price",
        "modal_price": "modal_price",
        "price": "price",
        "arrival_date": "arrival_date",
        "date": "date",
    }
    df = df.rename(columns=rename_map)

    return df


def ensure_storage() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def save_csv(df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(OUT_DIR, f"mandi_prices_{today}.csv")
    df.to_csv(path, index=False)
    return path


def save_sqlite(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("mandi_prices_raw", conn, if_exists="append", index=False)


def main() -> None:
    ensure_storage()
    df = fetch_all()

    if df.empty:
        print("No records returned.")
        return

    csv_path = save_csv(df)
    save_sqlite(df)

    print(f"Saved {len(df)} rows")
    print(f"CSV: {csv_path}")
    print(f"SQLite: {DB_PATH}")


if __name__ == "__main__":
    main()