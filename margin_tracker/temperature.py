"""市場全体の温度感（地合いのものさし）。

1. 信用評価損益率（市場全体、推定）
   JPX「銘柄別信用取引残高」の全銘柄データのうち、信用買残の金額（建値ベース）が大きい上位 TOP_N 銘柄について、
   「買残の株数 × その申込日の終値」と「買残の金額」を比べる。証券会社や日経が出す評価損益率とは計算元が違う推定値。
   一般によく言われる目安: -20%前後で底（投げ一巡）、-3%前後で天井（過熱）。
2. 日経平均のテクニカル
   25日・75日線からの乖離率、RSI(14)、ボリンジャーバンド(20日±2σ)の位置、13週線、52週高値からの下落率。
基準値はこのファイルの上の方にまとめてある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TOP_N = 500  # 信用買残の金額が大きい順に何銘柄で計算するか（上位500で全体の約8割）
HISTORY = Path("data/market_margin.csv")  # 市場全体の信用評価損益率の記録
PL_BOTTOM, PL_TOP = -0.20, -0.03  # 信用評価損益率: 底の目安・天井の目安
PL_COLD, PL_HOT = -0.15, -0.06  # これより下は「冷えている」、上は「熱い」寄り
DEV25_HOT, DEV25_COLD = 0.05, -0.05  # 日経の25日線乖離率
RSI_HOT, RSI_COLD = 70, 30


# ---------- 信用評価損益率 ----------
def margin_pl(day_df: pd.DataFrame, closes: dict[str, float]) -> tuple[float, float] | None:
    """day_df: その日の全銘柄の信用残（code, buy, buy_val）、closes: 銘柄コード → 終値。
    → (評価損益率, カバー率)。カバー率は、計算に使えた銘柄の買残金額 ÷ 全体の買残金額。"""
    d = day_df.dropna(subset=["buy", "buy_val"])
    d = d[d["buy_val"] > 0]
    total = float(d["buy_val"].sum())
    if total <= 0:
        return None
    use = d[d["code"].isin(closes)]
    if use.empty:
        return None
    cost = float(use["buy_val"].sum())
    value = float((use["buy"] * use["code"].map(closes)).sum())
    return value / cost - 1, cost / total


def top_codes(day_df: pd.DataFrame, n: int = TOP_N) -> list[str]:
    d = day_df.dropna(subset=["buy_val"])
    return list(d.sort_values("buy_val", ascending=False)["code"].head(n))


def update_history(days: dict[pd.Timestamp, pd.DataFrame], closes_fetcher, path: Path = HISTORY,
                   log=print) -> pd.DataFrame:
    """まだ記録していない申込日について、市場全体の信用評価損益率を計算して記録に足す。
    closes_fetcher(symbols) → {記号: 終値の Series}（market.spark_closes と同じ形）。"""
    hist = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=["date", "pl", "coverage"])
    have = set(pd.to_datetime(hist["date"]).dt.normalize())
    todo = {d: df for d, df in days.items() if pd.Timestamp(d).normalize() not in have}
    if not todo:
        return hist
    codes = sorted({c for df in todo.values() for c in top_codes(df)})
    try:
        series = closes_fetcher([f"{c}.T" for c in codes])
    except Exception as e:  # 取れなくてもレポートは出す
        log(f"信用評価損益率の株価取得に失敗: {e}")
        return hist
    rows = []
    for day, df in sorted(todo.items()):
        day = pd.Timestamp(day).normalize()
        closes = {}
        for c in top_codes(df):
            s = series.get(f"{c}.T")
            if s is None or s.empty:
                continue
            upto = s[s.index <= day]
            if not upto.empty and (day - upto.index[-1]).days <= 4:  # 申込日の終値（休日ならその前の営業日）
                closes[c] = float(upto.iloc[-1])
        res = margin_pl(df, closes)
        if res is not None:
            rows.append({"date": day, "pl": round(res[0], 5), "coverage": round(res[1], 4)})
    if rows:
        hist = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
        hist = hist.sort_values("date").drop_duplicates("date", keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        out = hist.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
    return hist


def pl_label(pl: float) -> str:
    if pl <= PL_BOTTOM:
        return "底の目安に到達"
    if pl >= PL_TOP:
        return "天井の目安に到達"
    if pl <= PL_COLD:
        return "冷えている"
    if pl >= PL_HOT:
        return "熱い"
    return "普通"


# ---------- 日経平均のテクニカル ----------
def rsi(close: pd.Series, period: int = 14) -> float:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain.iloc[-1] / last_loss))


@dataclass
class Gauge:
    name: str
    value: str
    note: str
    heat: int = 0  # +1 熱い（過熱）/ -1 冷えている（売られすぎ）/ 0 普通


def nikkei_gauges(df: pd.DataFrame | None) -> list[Gauge]:
    if df is None or len(df) < 80:
        return []
    c = df["Close"].dropna()
    last = float(c.iloc[-1])
    out = []
    dev25 = last / c.tail(25).mean() - 1
    heat = 1 if dev25 >= DEV25_HOT else -1 if dev25 <= DEV25_COLD else 0
    out.append(Gauge("25日乖離", f"{dev25:+.1%}", {1: "過熱", -1: "売られすぎ", 0: "普通"}[heat], heat))
    dev75 = last / c.tail(75).mean() - 1
    out.append(Gauge("75日乖離", f"{dev75:+.1%}", "線の上" if dev75 >= 0 else "線の下"))
    r = rsi(c)
    heat = 1 if r >= RSI_HOT else -1 if r <= RSI_COLD else 0
    out.append(Gauge("RSI", f"{r:.0f}", {1: "過熱", -1: "売られすぎ", 0: "普通"}[heat], heat))
    mid, sd = c.tail(20).mean(), c.tail(20).std()
    z = (last - mid) / sd if sd else 0.0
    heat = 1 if z >= 2 else -1 if z <= -2 else 0
    pos = "+2σ超え" if z >= 2 else "-2σ割れ" if z <= -2 else f"{z:+.1f}σ"
    out.append(Gauge("BB", pos, {1: "過熱", -1: "売られすぎ", 0: "バンド内"}[heat], heat))
    wk = c.resample("W-FRI").last().dropna()
    if len(wk) >= 13:
        ma13 = wk.tail(13).mean()
        out.append(Gauge("13週線", "上" if last >= ma13 else "下",
                         "上昇基調" if last >= ma13 else "怪しい（利確早め）"))
    high52 = float(c.tail(245).max())
    off = last / high52 - 1
    out.append(Gauge("高値から", f"{off:+.1%}", f"高値 {high52:,.0f}"))
    return out


@dataclass
class Temperature:
    gauges: list[Gauge] = field(default_factory=list)

    @property
    def summary(self) -> str:
        heat = sum(g.heat for g in self.gauges)
        if heat <= -2:
            return "冷えている（悲観）"
        if heat >= 2:
            return "熱い（楽観）"
        return "普通"


def build(margin_hist: pd.DataFrame | None, nikkei: pd.DataFrame | None) -> Temperature:
    t = Temperature()
    if margin_hist is not None and len(margin_hist):
        h = margin_hist.sort_values("date")
        last = h.iloc[-1]
        pl = float(last["pl"])
        heat = -1 if pl <= PL_COLD else 1 if pl >= PL_HOT else 0
        prev = h.iloc[-6] if len(h) >= 6 else (h.iloc[0] if len(h) >= 2 else None)
        t.gauges.append(Gauge("信用損益率", f"{pl:+.1%}", pl_label(pl), heat))
        if prev is not None:
            t.gauges.append(Gauge("└以前", f"{float(prev['pl']):+.1%}", f"{pd.Timestamp(prev['date']):%-m/%-d}申込"))
        t.gauges.append(Gauge("└範囲", f"{pd.Timestamp(last['date']):%-m/%-d}申込",
                              f"上位{TOP_N}・買残の{float(last['coverage']):.0%}"))
    nk = nikkei_gauges(nikkei)
    if nk:
        t.gauges.append(Gauge("**日経平均**", "", ""))
        t.gauges += nk
    return t


def fmt_none(v: float | None) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.1%}"
