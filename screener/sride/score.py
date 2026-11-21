"""フィルタリングとスコアリング."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODES, ScreenConfig


def _rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0〜1 のパーセンタイル。欠損は中立 (0.5)."""
    r = s.rank(pct=True, ascending=higher_is_better)
    return r.fillna(0.5)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["analysts"] = df["numberOfAnalystOpinions"].fillna(0).astype(int)
    df["turnover"] = df["averageDailyVolume3Month"] * df["regularMarketPrice"]
    df["net_cash"] = df["totalCash"].fillna(0) - df["totalDebt"].fillna(0)
    df["net_cash_ratio"] = df["net_cash"] / df["marketCap"]
    # ネットキャッシュ控除後PER: 時価総額から純現金を引いた「事業価値」で見たPER
    ev_ratio = (1 - df["net_cash_ratio"]).clip(lower=0.05)
    df["cash_adj_per"] = df["trailingPE"] * ev_ratio
    df["ev_ebitda"] = np.where(df["ebitda"] > 0,
                               (df["marketCap"] - df["net_cash"]) / df["ebitda"],
                               np.nan)
    # 純利益率が営業利益率を大きく上回る = 特別利益で PER が低く見えている可能性
    df["one_off_flag"] = (df["profitMargins"] > df["operatingMargins"].clip(lower=0) * 1.5 + 0.02)
    # 時価総額 / 純利益 で再計算した PER と Yahoo の PER が大きく乖離 = データ不整合
    per_calc = np.where(df["netIncomeToCommon"] > 0,
                        df["marketCap"] / df["netIncomeToCommon"], np.nan)
    df["per_calc"] = per_calc
    ratio = df["per_calc"] / df["trailingPE"]
    df["data_flag"] = ~ratio.between(0.6, 1.67) & ratio.notna()
    return df


FINANCIALS = {"銀行業", "証券、商品先物取引業", "保険業", "その他金融業"}


def apply_filters(df: pd.DataFrame, cfg: ScreenConfig) -> pd.DataFrame:
    m = (
        df["trailingPE"].between(0.1, cfg.max_per)
        & (df["priceToBook"] <= cfg.max_pbr)
        & (df["returnOnEquity"] >= cfg.min_roe)
        & (df["turnover"] >= cfg.min_turnover)
        & (df["analysts"] <= (cfg.max_analysts if cfg.max_analysts is not None else 10**6))
        & (~df["業種"].isin(FINANCIALS) if cfg.exclude_financials else True)
    )
    return df[m].copy()


def score(df: pd.DataFrame, mode: str = "fundamental") -> pd.DataFrame:
    df = df.copy()
    df["s_undiscovered"] = (
        _rank(df["analysts"], False) * 0.4
        + _rank(df["heldPercentInstitutions"], False) * 0.3
        + _rank(df["marketCap"], False) * 0.2
        + _rank(df["turnover"], False) * 0.1
    )
    df["s_value"] = (
        _rank(df["cash_adj_per"], False) * 0.35
        + _rank(df["priceToBook"], False) * 0.2
        + _rank(df["ev_ebitda"], False) * 0.25
        + _rank(df["net_cash_ratio"], True) * 0.2
    )
    df["s_quality"] = (
        _rank(df["returnOnEquity"]) * 0.3
        + _rank(df["returnOnAssets"]) * 0.15
        + _rank(df["operatingMargins"]) * 0.25
        + _rank(df["currentRatio"]) * 0.15
        + _rank(df["net_cash_ratio"]) * 0.05
        + _rank(df["heldPercentInsiders"]) * 0.10  # オーナー経営 = 利害一致
    )
    df["s_growth"] = (
        _rank(df["revenueGrowth"].clip(-0.5, 1.0)) * 0.5
        + _rank(df["earningsGrowth"].clip(-0.5, 2.0)) * 0.5
    )
    df["score"] = sum(df[f"s_{k}"] * w for k, w in MODES[mode].items()) * 100
    df.loc[df["one_off_flag"], "score"] *= 0.85  # 一過性利益の疑いは減点
    df.loc[df["data_flag"], "score"] *= 0.7      # データ不整合は大きく減点
    return df.sort_values("score", ascending=False)
