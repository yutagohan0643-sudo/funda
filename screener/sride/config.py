"""スクリーニング条件 (CLI 引数で上書き可)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScreenConfig:
    # --- 一次フィルタ (Yahoo バルクスクリーナー) ---
    min_mcap: float = 2e9          # 時価総額 下限 (円)
    max_mcap: float = 5e10         # 時価総額 上限 (円) = 500億円
    # --- 二次フィルタ (個別詳細データ) ---
    max_per: float = 20.0          # 実績PER 上限 (赤字は除外)
    max_pbr: float = 2.5
    min_roe: float = 0.08          # ROE 8%以上
    min_turnover: float = 5e6      # 3ヶ月平均売買代金 下限 (円/日) — 流動性
    max_analysts: int | None = None  # アナリスト数上限 (undiscovered モードは既定 2)
    mode: str = "fundamental"
    markets: tuple = ("プライム", "スタンダード", "グロース")
    exclude_financials: bool = True  # 銀行・証券等は純現金が歪むため除外
    workers: int = 4               # 詳細取得の並列数
    top: int = 30


# スコアの重み (合計 1.0)。--mode で切替
MODES = {
    # ファンダメンタルズのみ (既定)
    "fundamental": {"value": 0.35, "quality": 0.40, "growth": 0.25},
    # 未発掘度 (低カバレッジ・低機関保有・小型) を加味
    "undiscovered": {"undiscovered": 0.35, "value": 0.25, "quality": 0.25, "growth": 0.15},
}
MODE_LABELS = {
    "fundamental": "割安度35% + 質40% + 成長25%",
    "undiscovered": "未発掘度35% + 割安度25% + 質25% + 成長15%",
}
