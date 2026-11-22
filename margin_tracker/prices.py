"""株価データの取得（Yahoo Finance）と CSV 読み込み。"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import pandas as pd
import requests

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Dividend"]


def to_symbol(code: str) -> str:
    """'7203' や '130A' のような東証の銘柄コードを Yahoo の形式（7203.T）に直す。"""
    code = code.strip().upper()
    if re.fullmatch(r"\d{3}[0-9A-Z]", code):
        return f"{code}.T"
    return code


def fetch_yahoo(code: str, start: str | date, end: str | date | None = None) -> pd.DataFrame:
    """日足の株価と配当を取得する。

    OHLC は株式分割を調整した値。配当は1株あたりの金額で、権利落ち日の行に入る。
    """
    symbol = to_symbol(code)
    start_ts = int(pd.Timestamp(start).tz_localize("Asia/Tokyo").timestamp())
    end_dt = pd.Timestamp(end).tz_localize("Asia/Tokyo") + pd.Timedelta(days=1) if end else pd.Timestamp.now(tz="Asia/Tokyo")
    params = {
        "period1": start_ts,
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "div|split",
    }
    resp = requests.get(
        CHART_URL.format(symbol=symbol),
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    if resp.status_code == 404:
        raise ValueError(f"銘柄が見つかりません: {code}（{symbol}）")
    resp.raise_for_status()
    payload = resp.json()["chart"]
    if payload.get("error"):
        raise ValueError(f"データ取得エラー: {payload['error'].get('description')}")
    result = payload["result"][0]
    timestamps = result.get("timestamp")
    if not timestamps:
        raise ValueError(f"指定期間の株価データがありません: {symbol}")

    quote = result["indicators"]["quote"][0]
    tz = result["meta"].get("exchangeTimezoneName", "Asia/Tokyo")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=index,
    )
    df["Dividend"] = 0.0
    for div in (result.get("events", {}).get("dividends") or {}).values():
        day = pd.Timestamp(datetime.fromtimestamp(div["date"], tz=timezone.utc)).tz_convert(tz).normalize().tz_localize(None)
        if day in df.index:
            df.loc[day, "Dividend"] += float(div["amount"])

    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index.name = "Date"
    df.attrs["symbol"] = symbol
    df.attrs["name"] = result["meta"].get("longName") or result["meta"].get("shortName") or symbol
    first = result["meta"].get("firstTradeDate")
    if first:  # 取引開始日（新規上場銘柄なら上場日）
        df.attrs["first_trade"] = pd.Timestamp(first, unit="s", tz="UTC").tz_convert(tz).normalize().tz_localize(None)
    return df[COLUMNS]


def load_csv(path: str) -> pd.DataFrame:
    """Date, Open, High, Low, Close 列を持つ CSV を読む（Volume, Dividend は任意）。"""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    missing = {"Open", "High", "Low", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV に必要な列がありません: {', '.join(sorted(missing))}")
    for col in ("Volume", "Dividend"):
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.attrs["symbol"] = path
    df.attrs["name"] = path
    return df[COLUMNS]
