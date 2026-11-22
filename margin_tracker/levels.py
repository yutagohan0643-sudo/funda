"""株価の上の「抵抗になりやすい位置」と、下の「支えになりやすい位置」を、いくつかの根拠から探してまとめる。

抵抗（株価より上）の根拠:
  - 含み損の保有者が厚い価格帯の山（コスト分布の推定）… 買値まで戻ると「やれやれ売り」が出やすい
  - 信用買い方の平均建値（JPX の実データ）・保有者全体の平均コスト … 同じく売りが出やすい
  - 25日・75日移動平均線 … 多くの人が見ている線
  - 直近60営業日の戻り高値（その後10営業日以内に5%以上下げたもの）… 前回そこで売られた
  - 埋まっていない下落の窓（窓の上端） … 窓埋めの目標になりやすい
支え（株価より下）の根拠:
  - 含み益の保有者が厚い価格帯の山 … 安く買えた人が多く、買い直しや買い増しが入りやすい
  - 信用買い方の平均建値・保有者全体の平均コスト（株価より下にあるとき）
  - 25日・75日移動平均線
  - 直近半年で反発した安値（その後10営業日以内に5%以上戻したもの）… 前回そこで買われた
  - 埋まっていない上昇の窓（窓の下端） … 窓埋めで止まりやすい
  - フィボナッチ50%押し・ボリンジャー -2σ・EMA20 … ルール8ページの「サポートまで来た？」
  - 追証ゾーン（直近高値の -10〜20%）… 振り落としが出やすく、下げ切ると反発しやすい
近い価格（株価に近い側の端から MERGE_GAP 以内）の根拠は1つにまとめ、根拠の種類が多いほど強いとみなす。
同じ価格帯で2回以上反発した（止められた）ものは、根拠2つ分と数える。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cost_dist import CostDistribution

LOOKBACK_UP = 60  # 戻り高値を探す期間（営業日）
LOOKBACK_DOWN = 125  # 反発した安値を探す期間（約半年）
GAP_LOOKBACK = 60  # 窓を探す期間
SWING_WINDOW = 5  # 前後何日の中で一番高い（安い）なら山（谷）とするか
TURN_DAYS = 10  # 山（谷）のあと何日以内の動きを見るか
TURN_MIN = 0.05  # 山のあと5%以上下げた／谷のあと5%以上戻したものだけを数える
MA_PERIODS = (25, 75)
EMA_PERIOD = 20
BB_PERIOD, BB_SIGMA = 20, 2.0  # ボリンジャーバンド -2σ
FIB_LOOKBACK = 60  # フィボナッチ: 直近60日の上昇の波（安値→高値）の50%押し
FIB_HIGH_LOOKBACK = 20  # 波の高値は直近20日の高値
OIZOU_ZONE = (0.80, 0.90)  # 追証ゾーン: 直近20日の高値の -10〜20%（ルール8ページ）
MERGE_GAP = 0.02  # 株価に近い側の端から2%以内の根拠は1つにまとめる
MIN_DISTANCE = 0.003  # 株価との差が0.3%未満のものは「今いる位置」なので除く
BIG_BAND = 0.30  # 保有者の30%以上が集まる価格帯は、根拠2つ分の強さとみなす
TOP = 3

UP, DOWN = "up", "down"


@dataclass
class Point:
    price: float
    kind: str  # 根拠の種類（同じ種類は1つに数える）
    text: str
    weight: int = 1
    day: pd.Timestamp | None = None  # 戻り高値・反発した安値・窓の日付（並べ替え用）
    band: tuple[float, float, float] | None = None  # 厚い帯の (下限, 上限, 保有者の割合)


@dataclass
class Level:
    low: float
    high: float
    points: list[Point] = field(default_factory=list)
    side: str = UP  # UP = 抵抗（株価より上）、DOWN = 支え（株価より下）

    @property
    def score(self) -> int:
        best: dict[str, int] = {}
        for p in self.points:
            best[p.kind] = max(best.get(p.kind, 0), p.weight)
        swings = sum(p.kind == "swing" for p in self.points)
        if swings >= 2:
            best["swing"] = 2  # 同じ価格帯で2回以上反発した（止められた）
        return sum(best.values())

    @property
    def stars(self) -> str:
        n = max(1, min(3, self.score))
        return "★" * n + "☆" * (3 - n)

    @property
    def near(self) -> float:
        """株価に近い側の端。"""
        return self.low if self.side == UP else self.high

    @property
    def reasons(self) -> list[str]:
        """同じ種類の根拠は1つにまとめた説明。"""
        grouped: dict[str, list[Point]] = {}
        for p in self.points:
            grouped.setdefault(p.kind, []).append(p)
        out = []
        for kind, ps in grouped.items():
            ps = sorted(ps, key=lambda p: p.day if p.day is not None else pd.Timestamp.min)  # 日付の古い順
            days = [p.text for p in ps]
            if kind == "swing":
                name = "跳ね返された高値" if self.side == UP else "反発した安値"
                if len(days) >= 2:
                    name = f"{len(days)}回{'止められた高値' if self.side == UP else '反発した安値'}"
                out.append(f"{name}（{'・'.join(days[:2])}{' ほか' if len(days) > 2 else ''}）")
            elif kind == "gap":
                name = "下落の窓" if self.side == UP else "上昇の窓"
                out.append(f"{name}（{'・'.join(days[:2])}）")
            else:
                out.append(max(ps, key=lambda p: p.weight).text)
        return out

    @property
    def short_reasons(self) -> str:
        """表に入れる短い根拠。例: 厚い価格帯13%・EMA20"""
        names = {"avg": "平均コスト", "margin": "信用建値", "ma25": "25日線", "ma75": "75日線", "ema": "EMA20",
                 "bb": "-2σ", "fib": "フィボ50%", "oizou": "追証ゾーン"}
        grouped: dict[str, list[Point]] = {}
        for p in self.points:
            grouped.setdefault(p.kind, []).append(p)
        out = []
        for kind, ps in grouped.items():
            if kind == "band":
                for p in ps:  # 帯は範囲と割合を必ず書く
                    a, b, w = p.band
                    out.append(f"厚い帯 {a:,.0f}〜{b:,.0f}（{w:.0%}）")
            elif kind == "swing":
                base = "戻り高値" if self.side == UP else "反発安値"
                out.append(base + (f"×{len(ps)}" if len(ps) >= 2 else ""))
            elif kind == "gap":
                out.append("窓")
            else:
                out.append(names.get(kind, kind))
        return "・".join(out)

    def distance(self, close: float) -> float:
        return self.near / close - 1

    def price_text(self) -> str:
        if self.high - self.low < max(1.0, self.low * 0.003):
            return f"{self.near:,.0f}円"
        return f"{self.low:,.0f}〜{self.high:,.0f}円"


def _points(prices: pd.DataFrame | None, close: float, dist: CostDistribution | None, margin_cost: float,
            side: str) -> list[Point]:
    up = side == UP

    def beyond(x: float) -> bool:  # 探している側（上 or 下）にあるか
        return x > close if up else x < close

    out: list[Point] = []
    if dist is not None:
        for a, b, w in dist.bands:
            inside = (dist.prices >= a) & (dist.prices <= b) & ((dist.prices > close) if up else (dist.prices < close))
            if not inside.any():
                continue
            peak = float(dist.prices[inside][np.argmax(dist.weights[inside])])  # その側で一番厚い価格
            label = "含み損" if up else "含み益"
            out.append(Point(peak, "band", f"{label}の保有者が厚い（{a:,.0f}〜{b:,.0f}円に{w:.0%}）", band=(a, b, w), weight=
                             2 if w >= BIG_BAND else 1))
        if beyond(dist.avg_cost):
            out.append(Point(dist.avg_cost, "avg", "保有者全体の平均コスト"))
    if margin_cost == margin_cost and beyond(margin_cost):
        out.append(Point(margin_cost, "margin", "信用買い方の平均建値"))
    if prices is None or prices.empty:
        return out

    c = prices["Close"]
    for n in MA_PERIODS:
        if len(c) >= n:
            ma = float(c.tail(n).mean())
            if beyond(ma):
                out.append(Point(ma, f"ma{n}", f"{n}日移動平均"))
    if len(c) >= EMA_PERIOD:
        ema = float(c.ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])
        if beyond(ema):
            out.append(Point(ema, "ema", f"EMA{EMA_PERIOD}"))
    if not up and len(c) >= BB_PERIOD:
        tail = c.tail(BB_PERIOD)
        lower = float(tail.mean() - BB_SIGMA * tail.std())
        if beyond(lower):
            out.append(Point(lower, "bb", f"ボリンジャー -{BB_SIGMA:g}σ"))

    if not {"High", "Low"} <= set(prices.columns):  # 高値・安値がなければ、戻り高値・押し安値と窓は探さない
        return out
    if not up and len(prices) >= FIB_HIGH_LOOKBACK:
        hi_day = prices["High"].tail(FIB_HIGH_LOOKBACK).idxmax()
        hi = float(prices["High"].tail(FIB_HIGH_LOOKBACK).max())
        before = prices["Low"][prices.index <= hi_day].tail(FIB_LOOKBACK)
        lo = float(before.min())
        if hi > lo * 1.05:  # 5%以上の上昇の波があるときだけ
            fib = hi - (hi - lo) * 0.5
            if beyond(fib):
                out.append(Point(fib, "fib", f"フィボナッチ50%押し（{lo:,.0f}→{hi:,.0f}円の波）"))
        zone_hi, zone_lo = hi * OIZOU_ZONE[1], hi * OIZOU_ZONE[0]
        if beyond(zone_hi):
            out.append(Point(zone_hi, "oizou", f"追証ゾーン（高値{hi:,.0f}円の-10〜20%: {zone_lo:,.0f}〜{zone_hi:,.0f}円）"))
    # 期間の最初の日も前後の日と比べられるように、前の SWING_WINDOW 日分も読み込む
    recent = prices.tail((LOOKBACK_UP if up else LOOKBACK_DOWN) + SWING_WINDOW)
    highs = recent["High"].to_numpy(float)
    lows = recent["Low"].to_numpy(float)
    ext = highs if up else -lows  # 安値の谷は「符号を反対にした安値の山」として同じ方法で探す
    # 前の SWING_WINDOW 日がそろっていない最初の数日と、まだ山（谷）と決まらない最終日は除く
    for i in range(SWING_WINDOW, len(ext) - 1):
        window = ext[max(0, i - SWING_WINDOW): i + SWING_WINDOW + 1]
        before = ext[max(0, i - SWING_WINDOW): i]
        price = abs(float(ext[i]))
        # 同じ値が続く（横ばい）ときは最初の日だけを数える
        if not (ext[i] == window.max() and (before < ext[i]).all() and beyond(price)):
            continue
        after = slice(i + 1, i + 1 + TURN_DAYS)
        if up:
            move = lows[after].min(initial=price) / price - 1  # その後どれだけ下げたか
            ok = move <= -TURN_MIN
        else:
            move = highs[after].max(initial=price) / price - 1  # その後どれだけ戻したか
            ok = move >= TURN_MIN
        if ok:
            out.append(Point(price, "swing", f"{recent.index[i]:%-m/%-d}→{move:+.0%}", day=recent.index[i]))
    recent = prices.tail(GAP_LOOKBACK)
    highs = recent["High"].to_numpy(float)
    lows = recent["Low"].to_numpy(float)
    for i in range(1, len(recent)):
        if up:
            # 下落の窓: 前日の安値 > 当日の高値。その後の高値が前日の安値に届いていなければ未到達
            edge = lows[i - 1]
            open_gap = edge > highs[i] and highs[i + 1:].max(initial=-np.inf) < edge
        else:
            # 上昇の窓: 前日の高値 < 当日の安値。その後の安値が前日の高値まで下がっていなければ未到達
            edge = highs[i - 1]
            open_gap = edge < lows[i] and lows[i + 1:].min(initial=np.inf) > edge
        if open_gap and beyond(edge):
            out.append(Point(float(edge), "gap", f"{recent.index[i]:%-m/%-d}", day=recent.index[i]))
    return out


def _levels(prices, close, dist, margin_cost, side: str, top: int) -> list[Level]:
    up = side == UP
    pts = [p for p in _points(prices, close, dist, margin_cost, side) if abs(p.price / close - 1) >= MIN_DISTANCE]
    pts.sort(key=lambda p: p.price, reverse=not up)  # 株価に近い順
    levels: list[Level] = []
    for p in pts:
        if levels and abs(p.price / levels[-1].near - 1) <= MERGE_GAP:
            lv = levels[-1]
            lv.low, lv.high = min(lv.low, p.price), max(lv.high, p.price)
            lv.points.append(p)
        else:
            levels.append(Level(p.price, p.price, [p], side))
    return levels[:top]


def resistances(prices: pd.DataFrame | None, close: float, dist: CostDistribution | None,
                margin_cost: float, top: int = TOP) -> list[Level]:
    """株価の上にある抵抗を、近い順に top 個。"""
    return _levels(prices, close, dist, margin_cost, UP, top)


def supports(prices: pd.DataFrame | None, close: float, dist: CostDistribution | None,
             margin_cost: float, top: int = TOP) -> list[Level]:
    """株価の下にある支えを、近い順に top 個。"""
    return _levels(prices, close, dist, margin_cost, DOWN, top)
