"""スクリーナーの結果を Markdown（screener/README.md）にする。"""

from __future__ import annotations

import math

import pandas as pd

from . import core


def _pct(x: float, digits: int = 0) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


def _num(x: float, fmt: str) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _cap(x: float) -> str:
    if x is None or math.isnan(x):
        return "-"
    return f"{x / 1e8:,.0f}億" if x >= 1e8 else f"{x / 1e8:.1f}億"


def _margin(uni: pd.DataFrame, code: str) -> str:
    """信用買残の上場比（10%以上は需給が重い）。"""
    if code not in uni.index or pd.isna(uni.loc[code, "buy_ratio"]):
        return "-"
    r = float(uni.loc[code, "buy_ratio"])
    return f"{r * 100:.1f}%" + (" ⚠" if r >= 0.10 else "")


def build(results: list[core.Result], uni: pd.DataFrame, now: pd.Timestamp, sigma: float, days: int,
          scanned: int, min_turnover: float, upper: float | None = None,
          sub: tuple[float, int] | None = None) -> str:
    """upper: 予備軍のファイル（-sigma にはタッチしたが -upper には届かなかった銘柄）。
    sub: (σ, 銘柄数) 予備軍のファイルへのリンクを出す。"""
    passed = [r for r in results if r.passed]
    dropped = [r for r in results if r.fatal]
    unknown = [r for r in results if r.fund is None]
    span = "最新の営業日" if days == 1 else f"直近{days}営業日"

    reach = f"-{sigma:g}σ にタッチし、-{upper:g}σ には届かなかった" if upper else f"-{sigma:g}σ にタッチした"
    lines = [
        f"# 売られすぎスクリーナー（ボリンジャーバンド -{sigma:g}σ タッチ" + ("・予備軍）" if upper else "）"),
        "",
        f"更新: {now:%Y-%m-%d %H:%M}（日本時間）　対象 {scanned:,} 銘柄のうち、{span}に日足の安値が{reach}のは "
        f"**{len(results)} 銘柄**。致命的なファンダの問題がない **{len(passed)} 銘柄** が候補。",
        "",
    ]
    if upper:
        lines += [f"-{upper:g}σ までタッチした銘柄は [README.md](README.md)。ここは一歩手前の予備軍。", ""]
    if sub:
        lines += [f"-{sub[0]:g}σ にタッチした予備軍（-{sigma:g}σ には届かず、{sub[1]} 銘柄）は [2sigma.md](2sigma.md)。", ""]
    if not upper:
        lines += ["点数順のランキング（ファンダ・思惑・然るべき下げか不当な下げか）は [ranking.md](ranking.md)。", ""]
    if scanned and not upper and len(results) >= max(30, scanned * 0.02):
        lines += [f"> ⚠ タッチした銘柄が多い（{len(results)} 銘柄）。個別の売られすぎではなく、相場全体の急落の可能性。"
                  "地合いの落ち着き（VIX のピークアウトなど）を待つ方が安全。", ""]

    lines += [f"## 候補（-{sigma:g}σ タッチ × 致命的なファンダなし）", ""]
    if passed:
        lines += [
            "| 銘柄 | 終値 | タッチ日 | 安値σ | 今のσ | 戻り | 25日乖離 | RSI | 高値から | 出来高 | 評価 | PER | PBR | 時価総額 | 買残/上場 | 懸念 |",
            "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|",
        ]
        for r in sorted(passed, key=lambda r: r.touch.low_sigma):
            t, f = r.touch, r.fund
            lines.append(
                f"| {t.code} {t.name} | {t.close:,.0f} | {t.date:%m/%d} | {t.low_sigma:.1f} | {t.close_sigma:.1f} "
                f"| {_pct(t.rebound, 1)} | {_pct(t.dev25)} | {_num(t.rsi, '.0f')} | {_pct(t.from_high)} "
                f"| {_num(t.vol_ratio, '.1f')}倍 | {f.verdict or '-'} | {_num(f.per, '.1f')} | {_num(f.pbr, '.2f')} "
                f"| {_cap(f.market_cap)} | {_margin(uni, t.code)} | {' / '.join(r.concerns) or 'なし'} |"
            )
    else:
        lines.append("なし")
    lines.append("")

    lines += ["## 除外（致命的なファンダの問題あり）", ""]
    if dropped:
        lines += ["| 銘柄 | 終値 | 安値σ | 理由 |", "|---|--:|--:|---|"]
        for r in dropped:
            t = r.touch
            lines.append(f"| {t.code} {t.name} | {t.close:,.0f} | {t.low_sigma:.1f} | {' / '.join(r.fatal)} |")
    else:
        lines.append("なし")
    lines.append("")

    if unknown:
        lines += ["## ファンダ未確認（決算データが取れなかった）", "",
                  "タッチはしたが、決算データがないので判定していない。決算短信で確認してから検討する。", "",
                  "| 銘柄 | 終値 | 安値σ |", "|---|--:|--:|"]
        lines += [f"| {r.touch.code} {r.touch.name} | {r.touch.close:,.0f} | {r.touch.low_sigma:.1f} |" for r in unknown]
        lines.append("")

    lines += [
        "## 見方",
        "",
        f"- **タッチ**: その日の安値が、20日移動平均 − {sigma:g}×標準偏差（終値・母標準偏差）以下。"
        "正規分布なら -3σ の外は約0.1%、-2σ の外は約2.3%しか起きない。-3σ を「下げすぎ」、-2σ をその予備軍とみなす",
        "- **安値σ / 今のσ**: タッチした日の安値と、最新の終値が、移動平均から何σ離れているか",
        "- **戻り**: タッチした日の安値から最新の終値まで。大きいほど下ヒゲ・反発が出ている",
        "- **出来高**: タッチした日の出来高 ÷ 直近20日の平均。大きいと投げ売り（セリングクライマックス）の可能性",
        "- **評価・PER・PBR・時価総額**: 直近四半期を前年同期と比べた評価と、最新の株価で計算した値（technical の信用残ウォッチと同じ計算）",
        "- **買残/上場**: 信用買残 ÷ 上場株数（JPX）。⚠ は10%以上で、戻り売りが出やすい",
        f"- 平均売買代金が {min_turnover / 1e4:,.0f}万円/日 未満の銘柄は、売買しにくいので最初から外している",
        "",
        "### 致命的なファンダ（1つでもあれば除外）",
        "",
        "- 債務超過（自己資本がマイナス）",
        f"- 資金繰り: 直近1年が営業赤字で、現金が営業赤字の {core.FATAL_RUNWAY} 四半期分未満（増資・継続企業の注記のリスク）",
        f"- 大幅な希薄化: 株数が1年で {core.FATAL_DILUTION:.0%} 以上増えた（大型の増資・MSワラントなど）",
        f"- 売上の急減: 前年同期比 {core.FATAL_REV_DROP:.0%} 以下",
        "- 営業赤字が1年続き、さらに拡大している",
        "",
        "### 注意",
        "",
        "- 決算データは Yahoo Finance の四半期決算。小型株は誤りや反映の遅れがあるので、決算短信で確認する",
        "- 決算データでは見えない致命的な材料（不正会計・上場廃止の基準・監理銘柄・継続企業の注記・大きな訴訟など）は、"
        "下げた理由の適時開示（TDnet）で必ず確認する。-3σ は悪材料の直後に出やすい",
        "- -3σ は「下げすぎ」の目安で、そこが底とは限らない。反発（下ヒゲ・陽線）や出来高の減少を待つと安全",
        "",
        f"過去の結果は `data/screener/YYYYMMDD.csv` に保存（日付は株価の営業日）。",
        "",
    ]
    return "\n".join(lines)


TOP = 30  # ランキングに載せる銘柄数（出来高の区分ごと）
VOLUME_LINE = 500_000  # 直近20日の平均出来高（株）でランキングを2つに分ける境目


def _rank_table(ranked: list, sigma: float, top: int) -> list[str]:
    lines = [
        "| 順位 | 銘柄 | σ | 総合 | ファンダ | 思惑 | 反発 | 下げの判定 | 理由 | 好材料 | 決算 | PER | PBR | 時価総額 | 平均出来高 |",
        "|--:|---|--:|--:|--:|--:|--:|---|---|---|---|--:|--:|--:|--:|",
    ]
    for i, x in enumerate(ranked[:top], 1):
        t, f = x.result.touch, x.result.fund
        mark = "**" if t.low_sigma <= -sigma else ""
        lines.append(
            f"| {i} | {t.code} {t.name} | {mark}{t.low_sigma:.1f}{mark} | **{x.total}** | {x.fund_pts} | {x.hope_pts} | {x.rebound_pts} "
            f"| {x.verdict} | {' / '.join(x.reasons) or '-'} | {' / '.join(x.good_news) or '-'} | {f.verdict or '-'} "
            f"| {_num(f.per, '.1f')} | {_num(f.pbr, '.2f')} | {_cap(f.market_cap)} | {_vol(t.avg_volume)} |"
        )
    return lines + [""]


def _vol(v: float) -> str:
    if v is None or v != v:
        return "-"
    return f"{v:,.0f}株" if v < 1e4 else f"{v / 1e4:,.0f}万株"


def build_ranking(ranked: list, now: pd.Timestamp, sigma: float, days: int, tdnet_days: int, top: int = TOP,
                  volume_line: float = VOLUME_LINE) -> str:
    """ランキング（screener/ranking.md）。ranked は rank.rank() の結果（点数の高い順）。
    平均出来高が volume_line 株以上の銘柄と、それ未満の銘柄に分けて並べる。"""
    from . import rank as rk

    count = {v: sum(x.verdict == v for x in ranked) for v in (rk.UNJUSTIFIED, rk.UNCLEAR, rk.JUSTIFIED)}
    span = "最新の営業日" if days == 1 else f"直近{days}営業日"
    big = [x for x in ranked if x.result.touch.avg_volume >= volume_line]
    small = [x for x in ranked if not x.result.touch.avg_volume >= volume_line]
    line = f"{volume_line / 1e4:,.0f}万株"
    lines = [
        "# 売られすぎランキング（ファンダ × 思惑 × 下げの性質）",
        "",
        f"更新: {now:%Y-%m-%d %H:%M}（日本時間）　{span}にボリンジャーバンド -2σ 以上にタッチし、致命的なファンダの問題がない "
        f"**{len(ranked)} 銘柄** を点数順に並べた。直近20日の平均出来高で2つに分け、それぞれ上位 {top} 銘柄を載せる"
        "（全部は `data/screener/ranking_YYYYMMDD.csv`）。σ が太字なのは -3σ までタッチした銘柄。",
        "",
        f"- [平均出来高 {line}以上（{len(big)} 銘柄）](#平均出来高-{volume_line / 1e4:.0f}万株以上)",
        f"- [平均出来高 {line}未満（{len(small)} 銘柄）](#平均出来高-{volume_line / 1e4:.0f}万株未満)",
        "",
        f"下げの判定: **{rk.UNJUSTIFIED} {count[rk.UNJUSTIFIED]}** ／ {rk.UNCLEAR} {count[rk.UNCLEAR]} ／ "
        f"{rk.JUSTIFIED} {count[rk.JUSTIFIED]}",
        "",
        f"## 平均出来高 {line}以上",
        "",
        "売買が多く、まとまった株数でも売り買いしやすい銘柄。",
        "",
    ]
    lines += _rank_table(big, sigma, top)
    lines += [f"## 平均出来高 {line}未満", "", "出来高が少なめの銘柄。注目されていない分、未発掘の候補になりやすいが、板が薄く値が飛びやすい。", ""]
    lines += _rank_table(small, sigma, top)

    just = sorted([x for x in ranked if x.verdict == rk.JUSTIFIED], key=lambda x: -x.fund_pts)[:20]
    lines += [f"## {rk.JUSTIFIED}（悪材料が出ている。ファンダの点が高い順）", "",
              "割安に見えても、悪材料を織り込む途中の可能性がある。材料の中身を開示で確認してから。", "",
              "| 銘柄 | σ | 総合 | ファンダ | 平均出来高 | 悪材料 |", "|---|--:|--:|--:|--:|---|"]
    for x in just:
        t = x.result.touch
        lines.append(f"| {t.code} {t.name} | {t.low_sigma:.1f} | {x.total} | {x.fund_pts} | {_vol(t.avg_volume)} "
                     f"| {' / '.join(x.reasons)} |")
    lines.append("")

    lines += [
        "## 点数のつけ方（100点満点）",
        "",
        "- **ファンダ（40点）**: 直近決算の評価（増収増益 16・営業黒字に転換 13・減収増益 9・増収減益 5・減収減益 2）、"
        "PER（8倍未満 8・12倍未満 6・20倍未満 3）、PBR（0.7倍未満 8・1倍未満 6・1.5倍未満 3）、利益率の改善 4、実質無借金 4。"
        "懸念1つにつき -3（「最新決算が未反映かも」は -1）",
        "- **思惑（30点）**: タッチした日の出来高（平均の3倍 10・2倍 7・1.5倍 4）、売買の活発さ＝1日の売買代金÷時価総額"
        "（1%以上 8・0.5%以上 6・0.2%以上 3）、直近の好材料の開示（上方修正・増配・自社株買い・提携・受注など、1つ6点・最大12）",
        "- **反発（30点）**: 下げの性質（不当な下げ 15・要確認 6・然るべき下げ 0）、20日線までの戻り余地（10%以上 5・5%以上 3）、"
        "安値からの戻り＝下ヒゲ（3%以上 5・1.5%以上 3）、RSI（25以下 5・30以下 3）",
        f"- 信用買残が上場株数の{rk.HEAVY_MARGIN:.0%}以上なら -5（戻り売りが重い）",
        "",
        "## 下げの判定",
        "",
        f"- **{rk.JUSTIFIED}**: 直近{tdnet_days}日の適時開示（TDnet）に悪材料がある（下方修正・増資・新株予約権・減配・特損・不正・"
        "優待廃止など）。または、その間に決算を出し、中身が悪かった（減益・赤字転落・営業利益 -20%以下）",
        f"- **{rk.UNJUSTIFIED}**: 悪材料の開示がなく、直近{rk.WINDOW}日の下げの{rk.MARKET_SHARE:.0%}以上が地合い"
        f"（TOPIX かグロース250に、その銘柄のベータを掛けたもの）で説明できる。または、出来高を伴う窓あけの急落がなく、じりじり下げている",
        f"- **{rk.UNCLEAR}**: 開示は見当たらないのに出来高を伴って個別に急落した、または決算は良いのに急落した。"
        "TDnet 以外のニュース・会社予想の進捗・アナリストの評価などを確認する",
        "",
        "### 注意",
        "",
        "- 開示は表題の言葉だけで分けている（PDF の中身までは読まない）。「業績予想の修正」は表題では上か下か分からないので、"
        "株価が下げていれば下方の可能性として扱う",
        "- TDnet に出ない材料（新聞報道・格下げ・大株主の売り・業界のニュース）は拾えない。点数は絞り込みの目安で、売買の判断は自分で",
        "",
    ]
    return "\n".join(lines)
