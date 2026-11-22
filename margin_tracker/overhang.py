"""チャートの需給以外で、上値を押さえやすい要因（キーなしで分かる範囲）。

- IPO 後のロックアップ解除の目安（上場日から90日後・180日後）
- 株数の増加（増資・新株予約権・株式交換などによる希薄化。四半期決算の株数から）
- 信用買残の重さ（上場株数比）
新株予約権（MSワラント）・転換社債・売出し・大株主の売りは開示（TDnet・EDINET）にしか出ないので、手動確認の項目として残す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LOCKUP_DAYS = (90, 180)  # よくあるロックアップ期間（上場日から）
LOCKUP_WINDOW = 10  # 解除日の前後この日数に入ったら注目
RECENT_IPO_DAYS = 365  # 上場からこの日数以内なら IPO 銘柄として扱う
HEAVY_RATIO = 0.10  # 信用買残が上場株数のこれ以上なら重い
MANUAL = "新株予約権（MSワラント）・転換社債・売出し・大株主の売りは、適時開示（TDnet）と EDINET で手動確認"


@dataclass
class Overhang:
    items: list[str] = field(default_factory=list)  # 上値を押さえる要因
    signals: list[str] = field(default_factory=list)  # 今日の注目ポイントに出すもの


def check(prices: pd.DataFrame | None, today: pd.Timestamp, dilution: str | None = None,
          buy_ratio: float | None = None) -> Overhang:
    o = Overhang()
    first = prices.attrs.get("first_trade") if prices is not None else None
    if first is not None and (today - first).days <= RECENT_IPO_DAYS:
        opening = float(prices["Open"].iloc[0]) if prices.index[0] <= first + pd.Timedelta(days=3) else None
        parts = []
        for days in LOCKUP_DAYS:
            when = first + pd.Timedelta(days=days)
            left = (when - today).days
            state = f"あと{left}日" if left > 0 else ("今日" if left == 0 else f"{-left}日前に経過")
            parts.append(f"{days}日後 {when:%Y/%-m/%-d}（{state}）")
            if abs(left) <= LOCKUP_WINDOW:
                o.signals.append(f"ロックアップ解除の目安（上場{days}日後 {when:%-m/%-d}）が近い：VC・大株主の売りに注意")
        text = f"IPO後のロックアップ解除の目安（上場日 {first:%Y/%-m/%-d}）：" + "・".join(parts)
        if opening:
            text += f"。初値 {opening:,.0f}円"
        text += "。公開価格の1.5倍での解除条件などは目論見書で確認"
        o.items.append(text)
    if dilution:
        o.items.append(dilution)
    if buy_ratio is not None and buy_ratio == buy_ratio and buy_ratio >= HEAVY_RATIO:
        o.items.append(f"信用買残が上場株数の{buy_ratio:.1%}（戻り売り・返済売りの予備軍）")
    return o
