"""監視銘柄に上げる候補（昇格候補）を、売られすぎランキングから基準で選ぶ。

基準（どれも満たす銘柄を、総合点の高い順に最大10銘柄）:
- 下げの判定が「不当な下げ」（地合いや材料なしの下げで、会社の中身は悪くなっていない）
- 総合点（ファンダ・思惑・反発）が 65点以上
- 直近の決算が増益か黒字転換（「増益」「黒字」を含む）
- PER が 0〜15倍（直近4四半期の純利益から）
- 信用買残が上場株数の10%未満（分かる銘柄だけ。戻り売りが重くない）

選んだ銘柄は自動では監視銘柄にしない。人が TDnet で下げた理由と材料を確かめ、trade リポジトリの funda/ で基礎分析を作ってから足す。
"""

from __future__ import annotations

import math

import pandas as pd

MIN_TOTAL = 65
MAX_PER = 15
MAX_BUY_RATIO = 0.10
TOP = 10


def _ok_num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def row(x, uni: pd.DataFrame) -> dict:
    """ランキングの1銘柄（rank.Ranked）を、保存・判定に使う辞書にする。"""
    t, f = x.result.touch, x.result.fund
    ratio = None
    if t.code in uni.index and pd.notna(uni.loc[t.code, "buy_ratio"]):
        ratio = float(uni.loc[t.code, "buy_ratio"])
    return {
        "code": t.code, "name": t.name, "close": t.close, "low_sigma": round(t.low_sigma, 2),
        "avg_volume": round(t.avg_volume),
        "total": x.total, "fund": x.fund_pts, "hope": x.hope_pts, "rebound": x.rebound_pts, "verdict": x.verdict,
        "fund_verdict": f.verdict if f else "",
        "per": round(f.per, 1) if f and _ok_num(f.per) else None,
        "pbr": round(f.pbr, 2) if f and _ok_num(f.pbr) else None,
        "market_cap": round(f.market_cap) if f and _ok_num(f.market_cap) else None,
        "buy_ratio": round(ratio, 4) if ratio is not None else None,
        "concerns": " / ".join(x.result.concerns),
        "reasons": " / ".join(x.reasons), "good_news": " / ".join(x.good_news),
    }


def passes(r: dict) -> bool:
    if r["verdict"] != "不当な下げ" or r["total"] < MIN_TOTAL:
        return False
    if not any(k in (r["fund_verdict"] or "") for k in ("増益", "黒字")):
        return False
    if r["per"] is None or not 0 < r["per"] < MAX_PER:
        return False
    if r["buy_ratio"] is not None and r["buy_ratio"] >= MAX_BUY_RATIO:
        return False
    return True


def pick(rows: list[dict], top: int = TOP) -> list[dict]:
    return sorted((r for r in rows if passes(r)), key=lambda r: -r["total"])[:top]


def build(picked: list[dict], now: pd.Timestamp, total: int) -> str:
    lines = [
        "# 昇格候補（監視銘柄に上げるか検討する銘柄）",
        "",
        f"更新: {now:%Y-%m-%d %H:%M}（日本時間）　売られすぎランキングの {total:,} 銘柄から、下の基準をすべて満たす **{len(picked)} 銘柄**（最大{TOP}）。",
        "",
        "基準: 下げの判定が「不当な下げ」・総合点 "
        f"{MIN_TOTAL}点以上・直近の決算が増益か黒字転換・PER 0〜{MAX_PER}倍・信用買残が上場株数の{MAX_BUY_RATIO:.0%}未満。",
        "",
    ]
    if not picked:
        lines += ["今回は基準を満たす銘柄なし。", ""]
    else:
        lines += [
            "| 銘柄 | 終値 | σ | 総合 | 決算 | PER | PBR | 時価総額 | 買残/上場 | 下げの理由 | 好材料 | 懸念 |",
            "|---|--:|--:|--:|---|--:|--:|--:|--:|---|---|---|",
        ]
        for r in picked:
            cap = f"{r['market_cap'] / 1e8:,.0f}億" if r["market_cap"] else "-"
            ratio = f"{r['buy_ratio']:.1%}" if r["buy_ratio"] is not None else "-"
            lines.append(
                f"| {r['code']} {r['name']} | {r['close']:,.0f} | {r['low_sigma']:.1f} | **{r['total']}** | {r['fund_verdict']} "
                f"| {r['per']:.1f} | {r['pbr'] if r['pbr'] is not None else '-'} | {cap} | {ratio} "
                f"| {r['reasons'] or '-'} | {r['good_news'] or '-'} | {r['concerns'] or 'なし'} |"
            )
        lines.append("")
    lines += [
        "## 監視銘柄に上げる手順",
        "",
        "1. この表か、ファンダの候補（`candidates/` の ◎）から選ぶ",
        "2. TDnet で下げた理由・直近の材料・新株予約権などのリスクを確かめる（決算データだけでは悪材料が見えない）",
        "3. trade リポジトリの `funda/` で `コード_基礎.md` を作り、`funda/README.md` の表に足す（信用残ウォッチに自動で入る）",
        "4. trade の `valuation/stocks.json` に足して採点する。**60点以上なら監視を続け、45点未満なら表から外す**",
        "",
    ]
    return "\n".join(lines)
