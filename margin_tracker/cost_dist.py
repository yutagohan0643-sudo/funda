"""価格帯別の保有者分布（コスト分布）の推定。

moomoo などの「コスト分布」「チップ分布」と同じ考え方の推定モデル:
  - 毎日、出来高 ÷ 上場株式数 の割合（回転率）だけ、保有者が入れ替わったとみなす
  - 入れ替わった分は、その日の安値〜高値の間（典型価格 (高+安+終)/3 を頂点とする三角形）で買われたとする
  - それまでの保有者は、どの価格帯も同じ割合だけ減る
実際の保有データではないので、あくまで目安。

回転の効き具合（係数 decay）は銘柄ごとに自動で合わせる（calibrate）:
  出来高には同じ日に買って売るデイトレの分も入るので、出来高どおりに入れ替えると入れ替わりを多く見積もりやすい。
  JPX が公表する信用買い方の平均建値（買残の金額 ÷ 株数）は実データなので、
  モデルの平均コストがそれに一番近くなる係数を選ぶ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RATIO_HALF_STEP = 0.0005  # JPX の「上場比」は 0.1% 単位で丸められている
DECAY_CANDIDATES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5)  # 試す係数
DEFAULT_DECAY = 1.0
PREFER_DEFAULT_WITHIN = 0.02  # 標準の係数とのずれの差がこれ以内なら、標準の係数を使う
CALIBRATION_DAYS = 10  # 直近何日分の信用残で合わせるか


def estimate_listed_shares(hist: pd.DataFrame) -> float | None:
    """JPX の信用残（株数と上場比）から上場株式数を逆算する。

    上場比は 0.1% 単位で丸められているので、日ごと・買い売りごとに取りうる範囲を出し、
    すべての範囲が重なる区間の真ん中を使う。
    """
    lows, highs = [], []
    for side in ("buy", "sell"):
        if side not in hist or f"{side}_ratio" not in hist:
            continue
        for shares, ratio in zip(hist[side], hist[f"{side}_ratio"]):
            if not shares or ratio is None or not ratio == ratio or ratio < 0.001:
                continue
            lows.append(shares / (ratio + RATIO_HALF_STEP))
            highs.append(shares / (ratio - RATIO_HALF_STEP))
    if not lows:
        return None
    lo, hi = max(lows), min(highs)
    if lo <= hi:
        return (lo + hi) / 2
    return float(np.median([(a + b) / 2 for a, b in zip(lows, highs)]))  # 範囲が重ならないときは中央値


@dataclass
class CostDistribution:
    prices: np.ndarray  # 各価格帯の中心
    weights: np.ndarray  # 各価格帯の保有割合（合計 1）
    close: float
    bands: list[tuple[float, float, float]] = field(default_factory=list)  # 厚い価格帯 (下限, 上限, 割合)

    @property
    def avg_cost(self) -> float:
        return float(np.dot(self.prices, self.weights))

    @property
    def profit_ratio(self) -> float:
        """株価以下で買った保有者（含み益）の割合。"""
        return float(self.weights[self.prices <= self.close].sum())

    def quantile(self, q: float) -> float:
        cum = np.cumsum(self.weights)
        return float(self.prices[min(np.searchsorted(cum, q), len(self.prices) - 1)])

    def range_covering(self, share: float) -> tuple[float, float]:
        """保有者の真ん中 share（例 0.7 = 70%）が入る価格の範囲。"""
        tail = (1 - share) / 2
        return self.quantile(tail), self.quantile(1 - tail)


def estimate(prices: pd.DataFrame, listed_shares: float, bins: int = 150, decay: float = 1.0) -> CostDistribution:
    """日足（Open/High/Low/Close/Volume）と上場株式数からコスト分布を推定する。"""
    df = prices.dropna(subset=["High", "Low", "Close"])
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if hi <= lo:
        hi = lo * 1.01 + 1
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    chips = np.zeros(bins)

    for i, (h, l, c, v) in enumerate(zip(df["High"], df["Low"], df["Close"], df["Volume"].fillna(0))):
        day = _day_distribution(centers, float(l), float(h), (h + l + c) / 3)
        if i == 0:
            chips = day  # 最初の日は全員がその日の値段で持っているとする
            continue
        turnover = min(float(v) / listed_shares * decay, 1.0)
        chips = chips * (1 - turnover) + day * turnover

    chips = chips / chips.sum()
    dist = CostDistribution(centers, chips, float(df["Close"].iloc[-1]))
    dist.bands = _thick_bands(centers, chips)
    return dist


@dataclass
class Calibration:
    decay: float  # 選んだ係数
    error: float  # その係数での、信用買い方の平均建値とのずれ（平均。+ はモデルが高い）
    errors: dict[float, float]  # 係数ごとのずれ


def calibrate(prices: pd.DataFrame, listed_shares: float, hist: pd.DataFrame,
              candidates: tuple[float, ...] = DECAY_CANDIDATES) -> Calibration | None:
    """信用買い方の平均建値（実データ）に、モデルの平均コストが一番近くなる係数を選ぶ。

    hist は信用残の推移（列: date, buy, buy_val）。直近 CALIBRATION_DAYS 日分を使う。
    """
    rows = hist[(hist["buy"] > 0) & (hist["buy_val"] > 0)].tail(CALIBRATION_DAYS)
    targets = []
    for _, r in rows.iterrows():
        upto = prices[prices.index <= r["date"]]
        if len(upto) >= 20:
            targets.append((upto, r["buy_val"] / r["buy"]))
    if not targets:
        return None
    errors = {}
    for k in candidates:
        errs = [estimate(p, listed_shares, decay=k).avg_cost / cost - 1 for p, cost in targets]
        errors[k] = float(np.mean(errs))
    best = min(errors, key=lambda k: abs(errors[k]))
    if DEFAULT_DECAY in errors and abs(errors[DEFAULT_DECAY]) - abs(errors[best]) <= PREFER_DEFAULT_WITHIN:
        best = DEFAULT_DECAY
    return Calibration(best, errors[best], errors)


def _day_distribution(centers: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    if high - low < 1e-9:
        d = np.zeros_like(centers)
        d[np.abs(centers - low).argmin()] = 1.0
        return d
    peak = min(max(peak, low), high)
    d = np.where(
        centers < peak,
        (centers - low) / max(peak - low, 1e-9),
        (high - centers) / max(high - peak, 1e-9),
    )
    d = np.clip(d, 0, None) * ((centers >= low) & (centers <= high))
    if d.sum() == 0:  # 値幅が価格帯1つ分より狭い
        d = np.zeros_like(centers)
        d[np.abs(centers - peak).argmin()] = 1.0
    return d / d.sum()


def _thick_bands(centers: np.ndarray, weights: np.ndarray, top: int = 3,
                 min_share: float = 0.03) -> list[tuple[float, float, float]]:
    """保有者が集中している価格帯を上位 top 個。山ごとに、山の高さの半分以上ある範囲を1つの帯とする。
    保有者の割合が min_share 未満の小さな山は除く。"""
    smooth = np.convolve(weights, np.ones(5) / 5, mode="same")
    peaks = [i for i in range(1, len(smooth) - 1) if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]]
    peaks.sort(key=lambda i: smooth[i], reverse=True)
    bands, used = [], np.zeros(len(smooth), dtype=bool)
    for p in peaks:
        if used[p]:
            continue
        half = smooth[p] / 2
        a = p
        while a > 0 and smooth[a - 1] >= half and not used[a - 1]:
            a -= 1
        b = p
        while b < len(smooth) - 1 and smooth[b + 1] >= half and not used[b + 1]:
            b += 1
        used[a:b + 1] = True
        share = float(weights[a:b + 1].sum())
        if share < min_share:
            continue
        bands.append((float(centers[a]), float(centers[b]), share))
        if len(bands) == top:
            break
    return bands


def describe_bands(bands: list[tuple[float, float, float]]) -> str:
    return "<br>".join(f"{a:,.0f}〜{b:,.0f}円 ({w:.0%})" for a, b, w in bands) or "-"


def fmt_ratio(v: float | None) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"
