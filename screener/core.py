"""売られすぎスクリーナー：日足でボリンジャーバンド -3σ にタッチし、致命的なファンダの問題がない銘柄を探す。

流れ
  1. 対象: JPX の信用残データ（data/margin/daily/ の最新日）に載っている全銘柄。ETF・REIT など（上場比が空欄）は除く
  2. 一次ふるい: Yahoo の spark で全銘柄の終値をまとめて取り、終値が -2σ 以下（基準より1σ手前）まで来た銘柄だけ残す
  3. 判定: 残った銘柄の日足（始値・高値・安値・終値）を取り、安値が -3σ 以下になった日があるか見る（= タッチ）
  4. ファンダ: タッチした銘柄の四半期決算を見て、致命的な問題（債務超過・資金繰り・大幅な希薄化・売上の急減・
     赤字の拡大）がある銘柄を外す。問題はないが気になる点は「懸念」として出す

ボリンジャーバンドは 20日・終値・母標準偏差（多くのチャートソフトと同じ）。-3σ は、正規分布なら
約0.1%（1000日に1回）しか起きない水準なので「下げすぎ」の判定に使う。
判定の基準値はこのファイルの上の方にまとめてある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fundamentals import Fundamentals, evaluate

# ---- テクニカルの基準値 ----
BB_PERIOD = 20  # ボリンジャーバンドの期間（日）
BB_SIGMA = 3.0  # この σ の下限に安値がタッチしたら「下げすぎ」
SUB_SIGMA = 2.0  # 予備軍: この σ にはタッチしたが BB_SIGMA には届かなかった銘柄も別のファイルに出す
PRE_MARGIN = 1.0  # 一次ふるい: 終値が -(BB_SIGMA - 1.0)σ 以下まで来た銘柄だけ日足を詳しく見る（長い下ヒゲでのタッチも拾う。0.5 では取りこぼした）
LOOKBACK = 1  # 直近この営業日数のうちにタッチした銘柄を拾う（1 = 最新の日だけ）
NOT_STOCK = ("Investment Corporation", "Infrastructure Fund", "REIT")  # 名前にこれを含むものは外す
MIN_TURNOVER = 10_000_000  # 直近20日の平均売買代金がこれ未満（1000万円）の銘柄は、売買しにくいので外す

# ---- 致命的なファンダの基準値（どれか1つでも当てはまれば外す） ----
FATAL_DILUTION = 0.10  # 株数が1年で10%以上増えた（大型の増資・MSワラントなど）
FATAL_REV_DROP = -0.30  # 売上が前年同期比 -30% 以下（事業の急な縮小）
FATAL_RUNWAY = 8  # 直近1年が営業赤字で、現金が営業赤字の8四半期分未満（増資・継続企業の注記リスク）


@dataclass
class Touch:
    """-3σ タッチの判定結果（1銘柄）。"""

    code: str
    name: str
    date: pd.Timestamp  # タッチした日（複数日なら、いちばん深かった日）
    close: float  # 最新の終値
    low: float  # タッチした日の安値
    lower: float  # タッチした日の -3σ
    low_sigma: float  # タッチした日の安値が何σか（-3.4 など）
    close_sigma: float  # 最新の終値が何σか
    dev25: float  # 25日移動平均からの乖離率
    rsi: float  # RSI(14)
    from_high: float  # 52週高値からの下落率
    vol_ratio: float  # タッチした日の出来高 ÷ 直近20日の平均出来高
    turnover: float  # 直近20日の平均売買代金（円）
    avg_volume: float  # 直近20日の平均出来高（株）
    rebound: float  # タッチした日の安値から最新の終値までの戻り


@dataclass
class Result:
    touch: Touch
    fund: Fundamentals | None
    fatal: list[str] = field(default_factory=list)  # 致命的な問題（1つでもあれば外す）
    concerns: list[str] = field(default_factory=list)  # 致命的ではないが気になる点

    @property
    def passed(self) -> bool:
        return self.fund is not None and not self.fatal


def bands(close: pd.Series, period: int = BB_PERIOD) -> tuple[pd.Series, pd.Series]:
    """移動平均と標準偏差（母標準偏差 = ddof 0）。"""
    roll = close.rolling(period)
    return roll.mean(), roll.std(ddof=0)


def zscore(close: pd.Series, period: int = BB_PERIOD) -> pd.Series:
    """終値が移動平均から何σ離れているか。"""
    ma, sd = bands(close, period)
    return (close - ma) / sd.replace(0, np.nan)


def pre_screen(closes: dict[str, pd.Series], sigma: float = BB_SIGMA - PRE_MARGIN, lookback: int = LOOKBACK) -> list[str]:
    """終値だけで、直近 lookback 日に -sigma 以下まで来た銘柄を選ぶ（日足を詳しく取る候補）。"""
    out = []
    for code, c in closes.items():
        c = c.dropna()
        if len(c) < BB_PERIOD + lookback - 1:
            continue
        if (zscore(c).tail(lookback) <= -sigma).any():
            out.append(code)
    return out


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    if len(close) <= n or down.iloc[-1] == 0:
        return math.nan if len(close) <= n else 100.0
    return float(100 - 100 / (1 + up.iloc[-1] / down.iloc[-1]))


def find_touch(code: str, df: pd.DataFrame, sigma: float = BB_SIGMA, lookback: int = LOOKBACK,
               min_turnover: float = MIN_TURNOVER) -> Touch | None:
    """日足（Open/High/Low/Close/Volume）で、直近 lookback 日に安値が -sigma にタッチしたか。"""
    df = df.dropna(subset=["Low", "Close"])
    name = df.attrs.get("name", code)
    if any(w in name for w in NOT_STOCK):  # インフラファンド・REIT など（上場比があっても株ではない）
        return None
    if len(df) < BB_PERIOD + lookback - 1:
        return None
    c = df["Close"]
    ma, sd = bands(c)
    lower = ma - sigma * sd
    recent = df.tail(lookback)
    hit = recent.index[recent["Low"] <= lower.loc[recent.index]]
    if len(hit) == 0:
        return None
    vol = df["Volume"].fillna(0) if "Volume" in df else pd.Series(0.0, index=df.index)
    turnover = float((c * vol).tail(BB_PERIOD).mean())
    if turnover < min_turnover:
        return None
    z_low = (recent.loc[hit, "Low"] - ma.loc[hit]) / sd.loc[hit]
    day = z_low.idxmin()  # 複数日タッチしたら、いちばん深かった日
    avg_vol = vol.shift(1).rolling(BB_PERIOD).mean().loc[day]
    ma25 = c.rolling(25).mean().iloc[-1] if len(c) >= 25 else math.nan
    high52 = df["High"].tail(250).max() if "High" in df else c.tail(250).max()
    low = float(df.loc[day, "Low"])
    return Touch(
        code=code,
        name=name,
        date=day,
        close=float(c.iloc[-1]),
        low=low,
        lower=float(lower.loc[day]),
        low_sigma=float((low - ma.loc[day]) / sd.loc[day]),
        close_sigma=float(zscore(c).iloc[-1]),
        dev25=float(c.iloc[-1] / ma25 - 1) if ma25 == ma25 else math.nan,
        rsi=rsi(c),
        from_high=float(c.iloc[-1] / high52 - 1),
        vol_ratio=float(vol.loc[day] / avg_vol) if avg_vol and avg_vol == avg_vol else math.nan,
        turnover=turnover,
        avg_volume=float(vol.tail(BB_PERIOD).mean()),
        rebound=float(c.iloc[-1] / low - 1),
    )


def adjust_splits(q: pd.DataFrame | None, splits: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame | None:
    """株価は分割を調整済みなので、決算データの株数も分割後の株数にそろえる（決算期末より後の分割を掛ける）。"""
    if q is None or "shares" not in q or not splits:
        return q
    q = q.copy()
    for day, ratio in splits:
        if ratio > 0 and ratio != 1:
            q.loc[q.index < day, "shares"] *= ratio
    return q


def judge(q: pd.DataFrame | None, close: float, today: pd.Timestamp | None = None) -> tuple[Fundamentals | None, list[str], list[str]]:
    """決算データから (評価, 致命的な問題, 懸念) を出す。データがなければ評価は None（判定できない）。"""
    if q is None or q.empty or "revenue" not in q or "op" not in q:
        return None, [], []
    f = evaluate(q, close, today)
    fatal = []
    if "equity" in q and q["equity"].notna().any() and q["equity"].dropna().iloc[-1] <= 0:
        fatal.append("債務超過")
    op4 = q["op"].dropna().tail(4)
    loss_year = len(op4) == 4 and op4.sum() < 0
    if loss_year and "cash" in q and q["cash"].notna().any():
        burn = -op4.sum() / 4
        runway = q["cash"].dropna().iloc[-1] / burn
        if runway < FATAL_RUNWAY:
            fatal.append(f"資金繰り（現金は営業赤字の約{runway:.0f}四半期分）")
    if "shares" in q:
        sh = q["shares"].dropna()
        if len(sh) >= 5 and sh.iloc[-5] > 0 and sh.iloc[-1] / sh.iloc[-5] - 1 >= FATAL_DILUTION:
            fatal.append(f"大幅な希薄化（株数 {sh.iloc[-1] / sh.iloc[-5] - 1:+.0%}/年）")
    if not math.isnan(f.rev_yoy) and f.rev_yoy <= FATAL_REV_DROP:
        fatal.append(f"売上の急減（{f.rev_yoy:+.0%}・前年同期比）")
    if loss_year and f.verdict == "営業赤字が続く（拡大）":
        fatal.append("営業赤字が1年続き、拡大中")
    # 致命的なものと重なる懸念は出さない
    dup = ("債務超過", "現金", "株数")
    concerns = [c for c in f.concerns if not (fatal and c.startswith(dup))]
    return f, fatal, concerns
