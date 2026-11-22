"""監視銘柄の信用残レポート（Markdown＋グラフ画像）を作る。GitHub 上でそのまま読める形にする。"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .cost_dist import (
    DEFAULT_DECAY,
    CostDistribution,
    calibrate,
    describe_bands,
    estimate as estimate_cost,
    estimate_listed_shares,
)
from . import energy as en, fundamentals as fd, market as mk, overhang as oh, temperature as tp
from .levels import LOOKBACK_DOWN, Level, resistances, supports

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JST = ZoneInfo("Asia/Tokyo")
BUY_COLOR, SELL_COLOR, PRICE_COLOR = "#dc2626", "#2563eb", "#111827"
PROFIT_COLOR, LOSS_COLOR, AVG_COLOR = "#16a34a", "#f87171", "#7c3aed"


THEME_RE = re.compile(r"\[([^\]]*)\]")


def _watch_lines(path: str | Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        themes = [x.strip() for m in THEME_RE.findall(line) for x in m.replace("、", ",").split(",") if x.strip()]
        line = THEME_RE.sub("", line).strip()
        code, _, name = line.partition(" ")
        yield code.strip().upper(), name.strip() or code, themes


def read_watchlist(path: str | Path) -> list[tuple[str, str]]:
    """'2334 イオレ [テーマ, テーマ]' 形式の行を読む。# 以降はコメント。[ ] の中は関連する米国のテーマ。"""
    return [(code, name) for code, name, _ in _watch_lines(path)]


def read_themes(path: str | Path) -> dict[str, list[str]]:
    """監視銘柄ごとの、関連する米国のテーマ名（watchlist.txt の [ ] の中）。"""
    return {code: themes for code, _, themes in _watch_lines(path) if themes}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_num(v, pattern: str) -> str:
    return "-" if _missing(v) else pattern.format(v)


def _fmt_int(v) -> str:
    return _fmt_num(v, "{:,.0f}")


def _fmt_signed(v) -> str:
    return _fmt_num(v, "{:+,.0f}")


def _fmt_ratio(v) -> str:
    """信用倍率。売残がほぼない銘柄は桁が大きくなるので 100倍超 とまとめる。"""
    if _missing(v):
        return "-"
    return "100倍超" if v >= 100 else f"{v:.1f}倍"


def _fmt_pct(v) -> str:
    return _fmt_num(v, "{:+.1%}")


def summarize(hist: pd.DataFrame, prices: pd.DataFrame | None, indexes: dict | None = None,
              vix: pd.DataFrame | None = None) -> dict:
    """1銘柄の最新の指標をまとめる。hist は日付順。"""
    last = hist.iloc[-1]
    out = {
        "date": last["date"],
        "buy": last["buy"],
        "buy_chg": last["buy_chg"],
        "sell": last["sell"],
        "sell_chg": last["sell_chg"],
        "buy_ratio": last.get("buy_ratio"),
        "sell_ratio": last.get("sell_ratio"),
        "ratio": last["buy"] / last["sell"] if last["sell"] else math.nan,
        "avg_cost": last["buy_val"] / last["buy"] if last.get("buy_val") and last["buy"] else math.nan,
        "buy_5d": math.nan,
        "close": math.nan,
        "close_date": None,
        "pl": math.nan,
        "days_of_volume": math.nan,
    }
    if len(hist) >= 6:
        prev = hist.iloc[-6]["buy"]
        out["buy_5d"] = last["buy"] / prev - 1 if prev else math.nan
    if prices is not None and not prices.empty:
        out["close"] = float(prices["Close"].iloc[-1])
        out["close_date"] = prices.index[-1]
        if out["avg_cost"] == out["avg_cost"]:  # NaN でない
            out["pl"] = out["close"] / out["avg_cost"] - 1
        vol20 = prices["Volume"].tail(20).mean()
        if vol20:
            out["days_of_volume"] = last["buy"] / vol20
    out["listed"] = estimate_listed_shares(hist)
    out["dist"] = None
    needed = {"High", "Low", "Close", "Volume"}
    out["calib"] = None
    if prices is not None and len(prices) >= 20 and out["listed"] and needed <= set(prices.columns):
        if "buy_val" in hist:
            out["calib"] = calibrate(prices, out["listed"], hist)
        decay = out["calib"].decay if out["calib"] else DEFAULT_DECAY
        out["dist"] = estimate_cost(prices, out["listed"], decay=decay)
    out["resist"], out["support"] = [], []
    out["prices"] = prices
    if not _missing(out["close"]):
        out["resist"] = resistances(prices, out["close"], out["dist"], out["avg_cost"])
        out["support"] = supports(prices, out["close"], out["dist"], out["avg_cost"])
    out["rebound"] = None
    out["pressure"] = None
    if prices is not None and len(prices) >= 30 and needed <= set(prices.columns) and "Open" in prices:
        out["rebound"] = en.rebound(prices, (indexes or {}).get("グロース250"))
        out["pressure"] = en.pressure(prices, hist)
    return out


# ---- 「今日の注目」の判定基準（ここの数字を変えると注目の出方が変わる） ----
BUY_SURGE = 0.05  # 買残が前日比 +5% 以上 → 急増
BUY_DROP = -0.05  # 買残が前日比 -5% 以下 → 急減
DEEP_LOSS = -0.20  # 信用買い方の損益が -20% 以下 → 含み損が大きい
HEAVY_DAYS = 3.0  # 買残が出来高の3日分以上 → 重い
HEAVY_RATIO = 0.10  # 買残が上場株数の10%以上 → 需給が重い
THIN_SUPPORT = 0.05  # 含み益の保有者が 5% 未満 → 下に支えが薄い


def _man(v, signed: bool = False) -> str:
    """株数を「万株」で短く表す。"""
    if _missing(v):
        return "-"
    x = v / 10_000
    digits = 1 if abs(x) < 100 else 0
    return f"{x:{'+' if signed else ''},.{digits}f}万"


def _change_rate(now, chg) -> float:
    prev = now - chg
    return chg / prev if prev else math.nan


def signals(s: dict) -> list[str]:
    """1銘柄の「今日の注目」ポイント。"""
    out = []
    rate = _change_rate(s["buy"], s["buy_chg"])
    if rate == rate and rate >= BUY_SURGE:
        out.append(f"買残が急増 {rate:+.1%}（{_man(s['buy_chg'], True)}株）")
    elif rate == rate and rate <= BUY_DROP:
        out.append(f"買残が急減 {rate:+.1%}（{_man(s['buy_chg'], True)}株）")
    if s["pl"] == s["pl"] and s["pl"] <= DEEP_LOSS:
        out.append(f"信用買い方の含み損 {s['pl']:+.0%}（平均建値 {s['avg_cost']:,.0f}円）")
    ratio = s.get("buy_ratio")
    if ratio is not None and ratio == ratio and ratio >= HEAVY_RATIO:
        out.append(f"信用買残が上場株数の{ratio:.1%}（需給が重い）")
    if s["days_of_volume"] == s["days_of_volume"] and s["days_of_volume"] >= HEAVY_DAYS:
        out.append(f"買残が出来高の{s['days_of_volume']:.1f}日分（重い）")
    dist = s["dist"]
    if dist is not None:
        close = dist.close
        inside = [b for b in dist.bands if b[0] <= close <= b[1]]
        if inside:
            a, b, w = inside[0]
            out.append(f"株価が保有者の厚い価格帯 {a:,.0f}〜{b:,.0f}円（{w:.0%}）の中")
        if dist.profit_ratio < THIN_SUPPORT and not inside:
            out.append(f"含み益の保有者 {dist.profit_ratio:.0%}：下に保有者がほぼいない（支えが薄い）")
    return out


MARKS = "①②③④⑤"
CHART_RANGE = 0.40  # グラフに描く抵抗・支えは株価の ±40% 以内
RESIST_COLOR, RESIST_TEXT = "#f59e0b", "#b45309"  # 抵抗: 橙
SUPPORT_COLOR, SUPPORT_TEXT = "#14b8a6", "#0f766e"  # 支え: 青緑


def _no_support_text(prices: pd.DataFrame | None, close: float) -> str:
    text = "株価の下に目立つ支えなし"
    if prices is not None and "Low" in prices and len(prices):
        low = float(prices["Low"].tail(LOOKBACK_DOWN).min())
        if close <= low * 1.02:
            text += "（直近半年の安値圏。反発した実績のある安値がまだない）"
    return text


def level_lines(levels: list[Level], close: float) -> list[str]:
    """例: ①1,634円（+8%）★☆☆ 含み損の保有者が厚い（1,566〜1,702円に13%）"""
    return [
        f"{MARKS[i]} **{lv.price_text()}**（{lv.distance(close):+.0%}）{lv.stars}　{'・'.join(lv.reasons)}"
        for i, lv in enumerate(levels)
    ]


def _style(ax):
    ax.grid(alpha=0.22, lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=11)


def chart(code: str, name: str, hist: pd.DataFrame, prices: pd.DataFrame | None, s: dict, out: Path) -> None:
    """スマホで見やすい縦長のグラフ（画面幅に縮小されても読める文字の大きさにしてある）。
    上段左: 株価（半年）と平均コストの線 / 上段右: 価格帯別の保有者分布 / 下段: 信用残の棒グラフ（日ごと）
    """
    dist: CostDistribution | None = s["dist"]
    margin_cost = s["avg_cost"]
    fig = plt.figure(figsize=(5.4, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.7, 1.3], height_ratios=[3.1, 2], wspace=0.06, hspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2 = fig.add_subplot(gs[1, :])

    # 見出し
    rate = _change_rate(s["buy"], s["buy_chg"])
    parts = []
    if s["close"] == s["close"]:
        parts.append(f"株価 {s['close']:,.0f}円")
    if rate == rate:
        parts.append(f"買残 {_man(s['buy'])}株 {rate:+.1%}")
    fig.text(0.03, 0.992, f"{name}（{code}）", fontsize=18, fontweight="bold", va="top", color=PRICE_COLOR)
    fig.text(0.03, 0.952, "　".join(parts), fontsize=13, va="top", color="#4b5563")

    # 株価
    shown = None
    handles = []
    if prices is not None and not prices.empty:
        shown = prices["Close"].tail(130)
        ax1.plot(shown.index, shown, color=PRICE_COLOR, lw=1.8)
        ax1.scatter([shown.index[-1]], [shown.iloc[-1]], color=PRICE_COLOR, s=26, zorder=5)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%-m月"))
    if margin_cost == margin_cost:
        handles.append(ax1.axhline(margin_cost, color=BUY_COLOR, ls=(0, (5, 3)), lw=1.8,
                                   label=f"信用買い方の平均建値 {margin_cost:,.0f}円"))
    if dist is not None:
        handles.append(ax1.axhline(dist.avg_cost, color=AVG_COLOR, ls=(0, (1, 1.6)), lw=2.2,
                                   label=f"保有者全体の平均コスト {dist.avg_cost:,.0f}円"))
        step = dist.prices[1] - dist.prices[0] if len(dist.prices) > 1 else 1
        colors = [PROFIT_COLOR if x <= dist.close else LOSS_COLOR for x in dist.prices]
        axh.barh(dist.prices, dist.weights * 100, height=step, color=colors, linewidth=0)
        axh.axhline(dist.close, color=PRICE_COLOR, lw=1.2)
        axh.set_title(f"保有者の分布\n含み益 {dist.profit_ratio:.0%}", fontsize=12, color="#374151")
        axh.set_xlabel("%", fontsize=11, color="#6b7280")
        axh.xaxis.set_major_locator(plt.MaxNLocator(3))
        lo, hi = dist.quantile(0.01), dist.quantile(0.99)
        if shown is not None:
            lo, hi = min(lo, float(shown.min())), max(hi, float(shown.max()))
        pad = (hi - lo) * 0.05 or 1
        ax1.set_ylim(lo - pad, hi + pad)
    else:
        axh.text(0.5, 0.5, "保有者分布は\n計算できません", ha="center", va="center", fontsize=11,
                 color="#6b7280", transform=axh.transAxes)
    ax1.set_title("株価（直近半年）", fontsize=12, loc="left", color="#374151")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    axh.tick_params(labelleft=False)
    # 株価から遠すぎる抵抗・支えはグラフには描かない（縦軸が伸びて株価の線がつぶれるため。表には出す）
    resist = [lv for lv in s.get("resist") or [] if abs(lv.distance(s["close"])) <= CHART_RANGE]
    support = [lv for lv in s.get("support") or [] if abs(lv.distance(s["close"])) <= CHART_RANGE]
    if resist or support:
        ymin, ymax = ax1.get_ylim()
        if resist and max(lv.high for lv in resist) > ymax:  # グラフの外なら縦軸を広げて入れる
            ymax = max(lv.high for lv in resist) + (ymax - ymin) * 0.05
        if support and min(lv.low for lv in support) < ymin:
            ymin = min(lv.low for lv in support) - (ymax - ymin) * 0.05
        ax1.set_ylim(ymin, ymax)
        gap = (ymax - ymin) * 0.065  # 右端のラベル同士の最小間隔
        close = s["close"]
        for levels, color, text_color, prefix, direction in (
            (resist, RESIST_COLOR, RESIST_TEXT, "上", 1),
            (support, SUPPORT_COLOR, SUPPORT_TEXT, "下", -1),
        ):
            prev = close  # ラベルは株価から外側へ、重ならないように並べる
            for i, lv in enumerate(levels):
                bands = [p.band for p in lv.points if p.band is not None]
                for ax in (ax1, axh):
                    for a, b, _ in bands:  # 厚い帯は範囲を薄く塗り、上下の端に点線を引く（隣の帯と区別する）
                        ax.axhspan(a, b, color=color, alpha=0.12, lw=0, zorder=0)
                        for edge in (a, b):
                            ax.axhline(edge, color=color, lw=0.8, ls=(0, (2, 2)), alpha=0.8, zorder=1)
                    if lv.high > lv.low:
                        ax.axhspan(lv.low, lv.high, color=color, alpha=0.22, lw=0, zorder=0)
                    ax.axhline(lv.near, color=color, lw=1.8, zorder=1)
                limit = prev + direction * (gap if i else gap / 2)
                y = max(lv.near, limit) if direction > 0 else min(lv.near, limit)
                prev = y
                axh.annotate(f"{prefix}{MARKS[i]}{lv.near:,.0f}", xy=(1.0, lv.near), xycoords=("axes fraction", "data"),
                             xytext=(1.03, y), textcoords=("axes fraction", "data"), va="center", ha="left",
                             fontsize=11, color=text_color, fontweight="bold",
                             arrowprops=dict(arrowstyle="-", color=color, lw=0.8) if y != lv.near else None)
    for ax in (ax1, axh):
        _style(ax)
    if handles:  # 線の説明は見出しの下にまとめて置く（グラフに重ならないように）
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.925), frameon=False,
                   fontsize=11, handlelength=2.2)

    # 信用残（日ごとの棒グラフ。日付は独立した軸）
    h = hist.tail(15)
    x = list(range(len(h)))
    width = 0.4
    ax2.bar([i - width / 2 for i in x], h["buy"] / 10_000, width, color=BUY_COLOR, label="信用買残")
    ax2.bar([i + width / 2 for i in x], h["sell"] / 10_000, width, color=SELL_COLOR, label="信用売残")
    ax2.annotate(_man(h["buy"].iloc[-1]), (x[-1] - width / 2, h["buy"].iloc[-1] / 10_000), xytext=(0, 4),
                 textcoords="offset points", ha="center", fontsize=12, color=BUY_COLOR, fontweight="bold")
    step = max(1, len(h) // 5)
    ax2.set_xticks(x[::-1][::step][::-1])
    ax2.set_xticklabels([f"{d:%-m/%-d}" for d in h["date"].iloc[x[::-1][::step][::-1]]])
    ax2.set_ylabel("万株", fontsize=11, color="#6b7280")
    ax2.set_title("信用残の推移（申込日ごと）", fontsize=12, loc="left", color="#374151")
    ax2.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
    ax2.set_ylim(0, max(h["buy"].max(), h["sell"].max()) / 10_000 * 1.25 or 1)
    _style(ax2)
    ax2.grid(axis="x", visible=False)

    fig.subplots_adjust(left=0.13, right=0.8 if (resist or support) else 0.97, top=0.79, bottom=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _detail_table(s: dict) -> str:
    dist = s["dist"]
    rate = _change_rate(s["buy"], s["buy_chg"])
    sell_rate = _change_rate(s["sell"], s["sell_chg"])
    rows = [
        ("信用買残", f"**{_man(s['buy'])}株**（前日比 {_man(s['buy_chg'], True)}株 / {_fmt_pct(rate)}）"),
        ("信用買残の上場株数比", _fmt_num(s.get("buy_ratio"), "**{:.1%}**")),
        ("信用売残", f"{_man(s['sell'])}株（前日比 {_man(s['sell_chg'], True)}株 / {_fmt_pct(sell_rate)}）"
                     + ("" if _missing(s.get("sell_ratio")) else f"　上場株数の{s['sell_ratio']:.1%}")),
        ("信用倍率", _fmt_ratio(s["ratio"])),
        ("信用買い方の平均建値", "-" if _missing(s["avg_cost"]) else f"**{s['avg_cost']:,.0f}円**（損益 {_fmt_pct(s['pl'])}）"),
    ]
    if dist is not None:
        lo70, hi70 = dist.range_covering(0.7)
        rows += [
            ("保有者全体の平均コスト", f"**{dist.avg_cost:,.0f}円**（含み益の保有者 {dist.profit_ratio:.0%}）"),
            ("保有者の70%が入る価格帯", f"{lo70:,.0f}〜{hi70:,.0f}円"),
            ("保有者が厚い価格帯", describe_bands(dist.bands)),
        ]
    else:
        rows.append(("保有者分布", "計算できませんでした（株価データ不足、または上場株式数が逆算できない）"))
    resist = s.get("resist") or []
    rows.append(("抵抗になりやすい位置（上）", "<br>".join(level_lines(resist, s["close"])) if resist
                 else "株価の上に目立つ抵抗なし"))
    support = s.get("support") or []
    rows.append(("支えになりやすい位置（下）", "<br>".join(level_lines(support, s["close"])) if support
                 else _no_support_text(s.get("prices"), s["close"])))
    rows.append(("買残÷出来高", _fmt_num(s["days_of_volume"], "{:.1f}日分")))
    calib = s.get("calib")
    if dist is not None:
        if calib is not None:
            note = f"{calib.decay:g}（信用買い方の平均建値とのずれ {calib.error:+.0%}）"
            if abs(calib.error) > 0.10:
                note += "<br>ずれが大きい：信用買い方は保有者全体より高い（低い）値段で買った人に偏っている"
        else:
            note = f"{DEFAULT_DECAY:g}（標準。合わせ込みに使える信用残データなし）"
        rows.append(("回転の効き具合", note))
    return "| 項目 | 値 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def _yen(v: float) -> str:
    """金額を「億円」「万円」で短く。"""
    if _missing(v):
        return "-"
    if abs(v) >= 1e8:
        return f"{v / 1e8:,.1f}億円"
    return f"{v / 1e4:,.0f}万円"


def _fund_line(f: fd.Fundamentals) -> str:
    """カード用の1行。例: PER 20.0倍・PBR 4.1倍・時価総額 172億円　2026年4-6月期: 減収減益（営業利益 -29%）"""
    head = []
    head.append("PER " + ("赤字" if _missing(f.per) else f"{f.per:.1f}倍"))
    head.append("PBR " + _fmt_num(f.pbr, "{:.2f}倍"))
    if not _missing(f.market_cap):
        head.append(f"時価総額 {_yen(f.market_cap)}")
    text = "・".join(head)
    if f.verdict:
        text += f"　{f.period}: **{f.verdict}**（{_op_change(f)}）"
    return text


def _op_change(f: fd.Fundamentals) -> str:
    """営業利益の前年同期比。赤字が絡んで率が出せないときは、額を並べる。"""
    if not _missing(f.op_yoy):
        return f"営業利益 前年同期比 {f.op_yoy:+.0%}"
    prev = f.year_ago
    now = f.latest.get("op")
    if prev is None or _missing(now) or _missing(prev.get("op")):
        return f"営業利益 {_yen(now)}"
    return f"営業利益 {_yen(now)}、前年同期 {_yen(prev.get('op'))}"


def _fund_block(f: fd.Fundamentals | None) -> str:
    if f is None:
        return ""
    q = f.quarters
    trend = " → ".join(
        f"{d:%y}/{(d - pd.DateOffset(months=2)).month}-{d.month}月 {_yen(v)}" for d, v in q["op"].dropna().tail(5).items()
    ) if "op" in q else "-"
    last = f.latest
    rows = [
        ("PER・PBR", f"PER {'赤字' if _missing(f.per) else f'{f.per:.1f}倍'}（直近4四半期の純利益）・PBR {_fmt_num(f.pbr, '{:.2f}倍')}"),
        ("時価総額", _yen(f.market_cap)),
        ("直近決算", f"{f.period}：**{f.verdict or '前年同期と比べられない'}**"),
        ("売上", f"{_yen(last.get('revenue'))}（前年同期比 {_fmt_num(f.rev_yoy, '{:+.0%}')}）"),
        ("**営業利益**", f"**{_yen(last.get('op'))}**（{_op_change(f).replace('営業利益 ', '')}）"),
        ("営業利益率", _fmt_num(f.margin, "{:.1%}") + ("" if _missing(f.margin_diff) else f"（前年同期差 {f.margin_diff * 100:+.1f}pt）")),
        ("営業利益の推移", trend),
        ("良い点", "・".join(f.positives) or "特になし"),
        ("懸念", "・".join(f.concerns) or "特になし"),
    ]
    table = "| 項目 | 値 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (f"**ファンダ**\n\n{table}\n\n"
            "出典: Yahoo Finance の四半期決算（小型株は誤りや反映の遅れがあり得る。決算短信で確認）\n\n")


def _yen_short(v: float) -> str:
    """表のマス用の短い金額。例: 1.0億・1,070万・-7,814万"""
    if _missing(v):
        return "―"
    return f"{v / 1e8:,.1f}億" if abs(v) >= 1e8 else f"{v / 1e4:,.0f}万"


WJ = "\u2060"  # 単語結合子（目に見えない）。文字の間に入れると、そこでは改行されない


def nb(text: str) -> str:
    """スペース・「・」「、」「）」の後と「（」の直前以外では改行させない（スマホで「上/③」「-7,814/万」のように切れるのを防ぐ）。
    太字の ** の内側には入れない（Markdown が壊れるため）。"""
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and ch not in " ・、）" and nxt not in " ・（" and "*" not in (ch, nxt):
            out.append(WJ)
    return "".join(out)


def _table(rows: list[tuple[str, str]], head: tuple[str, str]) -> str:
    """2列の表（項目｜内容）。マスの中で強制改行はしない。"""
    lines = [f"| {head[0]} | {head[1]} |", "|---|---|"]
    lines += [f"| {nb(k)} | {nb(v)} |" for k, v in rows]
    return "\n".join(lines) + "\n\n"


def _ladder(s: dict) -> str:
    """抵抗・支えの「はしご」: 上に抵抗（高い順）、真ん中に株価、下に支え（高い順）。4列に分けて改行を防ぐ。"""
    resist = (s.get("resist") or [])[:3]
    support = (s.get("support") or [])[:3]
    if not resist and not support:
        return ""
    close = s["close"]

    def row(side: str, i: int, lv) -> str:
        why = lv.short_reasons.replace("厚い価格帯", "厚い帯")
        return f"| {nb(side + MARKS[i])} | {nb(f'{lv.near:,.0f}')} | {nb(_dist(lv.distance(close)))} | {nb(why)} |"

    lines = ["| 位置 | 価格 | 距離 | 根拠 |", "|---|--:|--:|---|"]
    lines += [row("上", i, lv) for i, lv in reversed(list(enumerate(resist)))]
    lines.append(f"| **株価** | **{close:,.0f}** | | |")
    lines += [row("下", i, lv) for i, lv in enumerate(support)] or ["| 下 | ― | | 支えなし |"]
    return "\n".join(lines) + "\n\n"


def _paired(left: list[tuple[str, str]], right: list[tuple[str, str]], heads: tuple[str, str, str, str]) -> str:
    """左右2組に並べた4列の表（横にはみ出さないように）。"""
    n = max(len(left), len(right))
    left = left + [("", "")] * (n - len(left))
    right = right + [("", "")] * (n - len(right))
    lines = ["| " + " | ".join(heads) + " |", "|---|--:|---|--:|"]
    lines += [f"| {nb(a)} | {nb(b)} | {nb(c)} | {nb(d)} |" for (a, b), (c, d) in zip(left, right)]
    return "\n".join(lines) + "\n\n"


def _trend_table(h: pd.DataFrame) -> str:
    """買残の推移: 左に3・2・1週前、右に直近3日（単位: 万株）。"""
    pts = margin_points(h)
    if not pts:
        return ""

    def val(r) -> str:
        return "―" if r is None else f"{r['buy'] / 1e4:,.0f}" if r["buy"] >= 1e6 else f"{r['buy'] / 1e4:,.1f}"

    weeks = [(label, val(r)) for label, r in pts if label.endswith("週前")]
    days = [(label, val(r)) for label, r in pts if not label.endswith("週前")]
    days[-1] = (days[-1][0], f"**{days[-1][1]}**")
    return _paired(weeks, days, ("買残", "万株", "直近", "万株"))


def _dist(v: float) -> str:
    """転換点までの距離。1%未満は小数1桁にする（-0% と出ないように）。"""
    return f"{v:+.1%}" if abs(v) < 0.01 else f"{v:+.0%}"


def comment(s: dict, grav: en.Gravity | None, sector: str, value: str, heavy: list[str],
            foot: en.Footprints | None = None) -> str:
    """銘柄ごとのひとこと: 引力 → セクター → 表面の評価 → 上値の重し → 下方向の圧力 → 次のポイント → 反発。"""
    out = []
    if grav is not None:
        g = {"上向き": "地合いの引力は上向き", "下向き": "地合いの引力は下向き", "横ばい": "地合いの引力は横ばい"}[grav.direction]
        if sector:
            same = (grav.direction == "上向き" and sector == "追い風") or (grav.direction == "下向き" and sector == "向かい風")
            g += f"、セクターも{sector}" if same else f"、セクターは{sector}"
        out.append(g + "。" + (f"{grav.turning}。" if grav.turning else ""))
    if value:
        out.append(f"PERなど表面上は{value.split('（')[0]}。")
    if heavy:
        out.append(f"上値の重しに{'・'.join(heavy)}。")
    pr = s.get("pressure")
    if pr is not None:
        out.append(f"需給は{pr.sentence}。")
    if foot is not None:
        names = {"出来高": "出来高の偏り", "OBV": "出来高の累積", "逆行": "地合いへの逆行", "下値": "下値で止まる",
                 "売残": "売残の増加", "買い集め": "大株主の買い集め", "値持ち": "大商い後の値持ち"}
        if len(foot.hits) >= 2:
            out.append(f"本尊の足跡{'あり' if len(foot.hits) >= 3 else 'が少し'}（{'・'.join(names[n] for n, _ in foot.hits)}）。")
        if foot.exits:
            out.append(f"手じまいの気配（{foot.exits[-1].split(' ', 1)[1]}）。")
    close = s["close"]
    sup = (s.get("support") or [None])[0]
    res = (s.get("resist") or [None])[0]
    if sup is not None and abs(sup.distance(close)) <= 0.03:
        out.append(f"下の転換点 {sup.near:,.0f}円（{_dist(sup.distance(close))}）に到達・接近。")
    elif res is not None and abs(res.distance(close)) <= 0.03:
        out.append(f"上の転換点 {res.near:,.0f}円（{_dist(res.distance(close))}）に接近。")
    elif sup is not None or res is not None:
        parts = []
        if sup is not None:
            parts.append(f"下{sup.near:,.0f}円（{_dist(sup.distance(close))}）")
        if res is not None:
            parts.append(f"上{res.near:,.0f}円（{_dist(res.distance(close))}）")
        out.append("次の転換点は" + "・".join(parts) + "。")
    rb = s.get("rebound")
    if rb is not None and rb.latest.startswith("逆行陽線"):
        out.append("地合いが悪い日に逆行陽線。")
    elif rb is not None and rb.latest in ("強い反発",):
        out.append("直近の日足で強い反発。")
    return "".join(out)


def _heavy(code: str, today: pd.Timestamp, fund: fd.Fundamentals | None, over: oh.Overhang) -> tuple[list[str], list[str]]:
    """上値の重し: 新株予約権・MSワラント・増資・売出し・大株主の異動（開示）、株数の増加、ロックアップ解除。
    → (表の行, ひとこと用の短い言葉)"""
    rows, short = [], []
    for day, kind, score, facts in en.overhangs(code, today):
        name = kind.split("・行使価額")[0].replace("（大株主の放出）", "").replace("の大量行使", "行使")
        fact = facts.split(" / ")[0] if facts else ""
        rows.append(f"{day:%-m/%-d} {name}" + (f"（{fact}）" if fact else ""))
        short.append(f"{name}（{day:%-m/%-d}）")
    for c in (fund.concerns if fund is not None else []):
        if c.startswith("株数"):
            rows.append(c)
            short.append("株数の増加")
    for item in over.items:
        if item.startswith("IPO後"):
            for part in item.split("：", 1)[1].split("。")[0].split("・"):
                days, rest = part.split("後 ", 1)
                date, state = rest.split("（", 1)
                date = date.split("/", 1)[1]  # 年を省く
                state = state.rstrip("）").replace("に経過", "経過")
                rows.append(f"ロック解除 {days} {date} {state}")
                if "あと" in state:
                    short.append(f"ロックアップ解除（{date}）")
    return rows, short


def _analysis_block(s: dict, h: pd.DataFrame, over: oh.Overhang, grav: en.Gravity | None,
                    themes: list[tuple[str, float, float]], fund: fd.Fundamentals | None, value: str,
                    mats: list[tuple[pd.Timestamp, str, float, str]], heavy_rows: list[str],
                    foot: en.Footprints | None = None) -> str:
    """分析の表（引力・セクター・妥当株価・新規材料・下方向の圧力・反発・信用）・転換点の表・買残の推移の表。
    項目名は5文字以内にして折り返しを防ぐ。"""
    rows = []
    if grav is not None:
        rows.append(("引力", f"**{grav.direction}**（{'・'.join(grav.reasons)}）"))
        if grav.turning:
            rows.append(("引力の変化", grav.turning))
    for name, c1, c5 in themes:
        rows.append(("セクター", f"{name} {c1:+.1%}（5日{c5:+.0%}）"))
    if fund is not None:
        rows.append(("PER・PBR", f"{'赤字' if _missing(fund.per) else f'{fund.per:.1f}倍'}・{_fmt_num(fund.pbr, '{:.2f}倍')}"))
    if value:
        rows.append(("表面の評価", f"**{value}**"))
    for item in heavy_rows:
        rows.append(("上値の重し", item))
    if not heavy_rows:
        start = en.record_start()
        rows.append(("上値の重し", f"記録上なし（{start:%-m/%-d}〜）" if start is not None else "記録なし"))
    if mats:
        for day, kind, score, facts in mats[:3]:
            sign = "+" if score > 0 else ""
            rows.append(("新規材料", f"{day:%-m/%-d} {kind}（{sign}{score:.0f}）"))
    else:
        rows.append(("新規材料", "直近30日なし"))
    pr = s.get("pressure")
    if pr is not None:
        rows.append(("下方向圧力", f"**{pr.level}**"))
        rows.append(("中身", pr.meaning))
        rows.append(("5日の変化", pr.breakdown))
    if foot is not None:
        rows.append(("本尊の気配", f"**{foot.level}**"))
        for name, detail in foot.hits:
            rows.append((f"└{name}", detail))
        for e in foot.exits:
            rows.append(("└手じまい", e))
    rb = s.get("rebound")
    if rb is not None and rb.marks:
        rows.append(("反発サイン", f"**{rb.latest}**"))
        rows.append(("直近3日", rb.days_text))
    rate = _change_rate(s["buy"], s["buy_chg"])
    rows.append(("信用買残", f"{_man(s['buy'])}株（{_fmt_pct(rate)}）"))
    if not _missing(s.get("buy_ratio")):
        rows.append(("上場株数比", f"**{s['buy_ratio']:.1%}**"))
    if not _missing(s["avg_cost"]):
        rows.append(("信用建値", f"{s['avg_cost']:,.0f}円（{_fmt_pct(s['pl'])}）"))
    if s.get("dist") is not None:
        rows.append(("含み益保有", f"{s['dist'].profit_ratio:.0%}"))
    return _table(rows, ("分析", "　")) + _ladder(s) + _trend_table(h)


def _fund_short(f: fd.Fundamentals | None) -> str:
    """ファンダの表と、営業利益の推移の表。"""
    if f is None:
        return _table([("決算データ", "なし")], ("ファンダ", "　"))
    rows = [
        ("PER", "赤字" if _missing(f.per) else f"{f.per:.1f}倍"),
        ("PBR", _fmt_num(f.pbr, "{:.2f}倍")),
        ("時価総額", _yen(f.market_cap)),
    ]
    d = f.quarters.index[-1]
    period = f"{d:%y}/{(d - pd.DateOffset(months=2)).month}-{d.month}月"
    rows.append(("直近決算", f"{period} **{f.verdict}**" if f.verdict else period))
    last, prev = f.latest, f.year_ago
    op = _yen_short(last.get("op"))
    if not _missing(f.op_yoy):
        rows.append(("営業利益", f"{op}（{f.op_yoy:+.0%}）"))
    elif prev is not None:
        rows.append(("営業利益", f"{op}（前年 {_yen_short(prev.get('op'))}）"))
    else:
        rows.append(("営業利益", op))
    if not _missing(f.rev_yoy):
        rows.append(("売上", f"{_yen_short(last.get('revenue'))}（{f.rev_yoy:+.0%}）"))
    for c in f.concerns:
        rows.append(("懸念", c))
    out = _table(rows, ("ファンダ", "　"))
    if "op" in f.quarters:
        ops = [(f"{d:%y}/{(d - pd.DateOffset(months=2)).month}-{d.month}", _yen_short(v))
               for d, v in f.quarters["op"].dropna().tail(5).items()]
        out += _table(ops, ("期", "営業利益")).replace("|---|---|", "|---|--:|")
    return out


def _overhang_block(o: oh.Overhang) -> str:
    lines = ["**上値を押さえる要因（チャート以外）**\n"]
    lines += [f"- {x}" for x in o.items] or ["- 自動で分かる範囲では特になし"]
    lines.append(f"- {oh.MANUAL}")
    return "\n".join(lines) + "\n\n"


def _theme_text(market: mk.Market | None, names: list[str]) -> tuple[str, list[str]]:
    """カードに出す関連テーマの騰落と、大きく動いたときの注目ポイント。"""
    if not names:
        return "", []
    if market is None or not (market.themes or market.sectors):
        return "（米国のデータなし）", []
    parts, ups, downs = [], [], []
    for key in names:
        found = mk.find_theme(market, key)
        if found is None:
            parts.append(f"{key}（テーマ名が見つからない）")
            continue
        n, _, c, c5 = found
        color = "🟢" if c >= 0 else "🔴"
        parts.append(f"{color}{n} **{c:+.1%}**（5日 {c5:+.1%}）")
        if c >= mk.THEME_MOVE:
            ups.append(f"{n} {c:+.1%}")
        elif c <= -mk.THEME_MOVE:
            downs.append(f"{n} {c:+.1%}")
    sig = []
    if ups:
        sig.append("前夜の米国で関連テーマが大きく上昇：" + "・".join(ups))
    if downs:
        sig.append("前夜の米国で関連テーマが大きく下落：" + "・".join(downs))
    return "　".join(parts), sig


def _fmt_close(r: mk.Reading) -> str:
    if r.ind.kind == "rate":
        return f"{r.close:.2f}%"
    if r.ind.kind in ("vix", "fx", "commodity"):
        return f"{r.close:,.2f}"
    return f"{r.close:,.0f}" if r.close >= 1000 else f"{r.close:,.1f}"


def _fmt_change(r: mk.Reading, v: float) -> str:
    if _missing(v):
        return "-"
    return f"{v:+.2f}pt" if r.ind.kind == "rate" else f"{v:+.1%}"


LABEL_MARK = {mk.TAIL: "🟢追い風", mk.NEUTRAL: "⚪中立", mk.HEAD: "🔴向かい風", mk.WATCH: "🟡注意"}


def ranking_chart(rows: list[tuple[str, float, float]], title: str, out: Path) -> None:
    """騰落ランキング（横棒＝前日比）。右側に前日比と5日比の数値を並べる。rows: (名前, 前日比, 5日比) を上から順に。"""
    if not rows:
        return
    rows = rows[::-1]
    n = len(rows)
    height = 1.4 + 0.34 * n
    names = [name for name, _, _ in rows]
    day = [c * 100 for _, c, _ in rows]
    fig, ax = plt.subplots(figsize=(5.4, height))
    ax.barh(names, day, color=[PROFIT_COLOR if v >= 0 else BUY_COLOR for v in day], height=0.62)
    ax.axvline(0, color="#9ca3af", lw=1)
    lim = max(max(abs(v) for v in day) * 1.1, 1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, n + 0.4)  # 一番上に列の見出しを置く余白
    cols = ((1.42, "前日比"), (1.95, "5日比"))
    for x, head in cols:
        ax.annotate(head, xy=(x, n - 0.1), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=10, color="#6b7280")
    for i, (_, c, c5) in enumerate(rows):
        ax.annotate(f"{c:+.1%}", xy=(cols[0][0], i), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=11, color=PROFIT_COLOR if c >= 0 else BUY_COLOR, fontweight="bold")
        ax.annotate(f"{c5:+.1%}", xy=(cols[1][0], i), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=11, color="#374151")
    fig.text(0.03, 1 - 0.15 / height, title, fontsize=14, fontweight="bold", va="top", color="#111827")
    ax.set_xlabel("%", fontsize=10, color="#6b7280")
    _style(ax)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.37, right=0.65, top=1 - 0.65 / height, bottom=0.55 / height)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def temperature_section(temp: tp.Temperature | None) -> str:
    """市場全体の温度感（信用評価損益率・日経平均のテクニカル）。"""
    if temp is None or not temp.gauges:
        return ""
    lines = [f"**温度感：{temp.summary}**\n", "| ものさし | 値 | 目安 |", "|---|--:|---|"]
    lines += [f"| {g.name if g.name.startswith('**') else nb(g.name)} | {nb(g.value)} | {nb(g.note)} |" for g in temp.gauges]
    return "\n".join(lines) + "\n"


def market_section(m: mk.Market | None, chart_dir: Path, temp: tp.Temperature | None = None) -> str:
    if m is None or not m.readings:
        return ""
    grav = en.gravity(m.series) if m.series else None
    lines = [
        "## 地合い\n",
        (f"**引力：{grav.direction}**（{'・'.join(grav.reasons)}）" + (f"　{grav.turning}" if grav.turning else "") + "\n")
        if grav is not None else "",
        temperature_section(temp),
        f"**{mk.overall(m)}**　🟢追い風 {m.count(mk.TAIL)}・🔴向かい風 {m.count(mk.HEAD)}・🟡注意 {m.count(mk.WATCH)}・⚪中立 {m.count(mk.NEUTRAL)}\n",
    ]
    notes, below_ma = [], []
    for region in ("米国", "日本", "その他"):
        rs = [r for r in m.readings if r.ind.region == region]
        if not rs:
            continue
        lines += [f"**{region}**（{max(r.date for r in rs):%-m/%-d} 終値）\n", "| 指標 | 終値 | 前日比 | |", "|---|--:|--:|:-:|"]
        for r in rs:
            lines.append(f"| {r.ind.name} | {_fmt_close(r)} | {_fmt_change(r, r.change)} | {LABEL_MARK[r.label][0]} |")
            extra = [x for x in r.note.split("／") if x and x not in ("25日線の上", "25日線の下")]
            if "25日線の下" in r.note:
                below_ma.append(r.ind.name)
            if extra:
                notes.append(f"{r.ind.name}：{'／'.join(extra)}")
        lines.append("")
    if notes or below_ma:
        lines += [f"- {n}" for n in notes]
        if below_ma:
            lines.append(f"- 25日線の下：{'・'.join(below_ma)}")
        lines.append("")
    if m.sectors:
        date = max(x[4] for x in m.sectors)
        ranking_chart([(n, c, c5) for _, n, c, c5, _ in m.sectors], f"米国セクター別の騰落（{date:%-m/%-d}）",
                      chart_dir / "market_sectors.png")
        lines.append("![米国セクター別の騰落](charts/market_sectors.png)\n")
    if m.themes:
        date = max(x[4] for x in m.themes)
        ranking_chart([(n, c, c5) for _, n, c, c5, _, _ in m.themes], f"米国テーマ・業種別の騰落（{date:%-m/%-d}）",
                      chart_dir / "market_themes.png")
        top = "・".join(f"{n} {c:+.1%}" for _, n, c, _, _, _ in m.themes[:3])
        bottom = "・".join(f"{n} {c:+.1%}" for _, n, c, _, _, _ in m.themes[-3:][::-1])
        lines += [f"- 強い: {top}", f"- 弱い: {bottom}\n",
                  "![米国テーマ・業種別の騰落](charts/market_themes.png)\n"]
    if m.sectors or m.themes:
        lines += ["<details><summary>セクター・テーマの数値（分類別）</summary>\n",
                  "| 分類 | 名前（ETF） | 前日比 | 5日比 |", "|---|---|--:|--:|"]
        lines += [f"| セクター | {n}（{sym}） | {c:+.1%} | {c5:+.1%} |" for sym, n, c, c5, _ in m.sectors]
        order = {g: i for i, g in enumerate(dict.fromkeys(g for *_, g in mk.THEMES))}
        for sym, n, c, c5, _, g in sorted(m.themes, key=lambda x: (order.get(x[5], 99), -x[2])):
            lines.append(f"| {g} | {n}（{sym}） | {c:+.1%} | {c5:+.1%} |")
        lines += ["\n</details>\n"]
    if m.errors:
        lines.append("取得できなかった指標: " + "、".join(e.split(":")[0] for e in m.errors) + "\n")
    return "\n".join(lines)


RECENT_DAYS = 3  # 直近何日分を出すか
WEEKS_BACK = (1, 2, 3)  # 何週前と比べるか


def margin_points(h: pd.DataFrame) -> list[tuple[str, pd.Series | None]]:
    """信用残の推移を見る時点: 3週前・2週前・1週前・直近3日（古い順）。データがない時点は None。

    「n週前」は、最新の申込日の7n日前か、それより前で一番近い日のデータ。
    """
    h = h.sort_values("date")
    if h.empty:
        return []
    latest = h["date"].iloc[-1]
    recent = h.tail(RECENT_DAYS)
    out: list[tuple[str, pd.Series | None]] = []
    for w in sorted(WEEKS_BACK, reverse=True):
        before = h[(h["date"] <= latest - pd.Timedelta(days=7 * w)) & (h["date"] < recent["date"].iloc[0])]
        out.append((f"{w}週前", before.iloc[-1] if not before.empty else None))
    out += [(f"{r['date']:%-m/%-d}", r) for _, r in recent.iterrows()]
    return out


def _trend_line(h: pd.DataFrame) -> str:
    """カード用: 3週前 ― ・ 2週前 ― ・ 1週前 1,120万 ｜ 11/17 1,099万 → 11/18 1,173万 → 11/19 1,280万"""
    pts = margin_points(h)
    if not pts:
        return ""
    weeks = [(label, r) for label, r in pts if label.endswith("週前")]
    days = [(label, r) for label, r in pts if not label.endswith("週前")]
    left = "・".join(f"{label} {'―' if r is None else _man(r['buy'])}" for label, r in weeks)
    right = " → ".join(f"{label} {_man(r['buy'])}" for label, r in days[:-1])
    last_label, last = days[-1]
    right = (right + " → " if right else "") + f"**{last_label} {_man(last['buy'])}**"
    return f"{left}　｜　{right}"


def _history_table(h: pd.DataFrame) -> str:
    pts = margin_points(h)
    if not pts:
        return ""
    now = pts[-1][1]["buy"]
    day_names = ["最新", "前営業日", "2営業日前"]
    n_days = sum(not label.endswith("週前") for label, _ in pts)
    names = [label for label, _ in pts if label.endswith("週前")] + day_names[:n_days][::-1]
    lines = ["| 時点 | 申込日 | 買残 | 売残 | そこから今まで |", "|---|---|--:|--:|--:|"]
    for label, (_, r) in zip(names, pts):
        if r is None:
            lines.append(f"| {label} | ― | ― | ― | データなし |")
            continue
        change = "（最新）" if r is pts[-1][1] else _fmt_pct(now / r["buy"] - 1 if r["buy"] else math.nan)
        lines.append(f"| {label} | {r['date']:%-m/%-d} | {_man(r['buy'])} | {_man(r['sell'])} | {change} |")
    return "\n".join(lines)


def build(watchlist: list[tuple[str, str]], hist_all: pd.DataFrame, prices: dict[str, pd.DataFrame | None],
          out_dir: Path, now: datetime | None = None, market: mk.Market | None = None,
          themes: dict[str, list[str]] | None = None, funds: dict[str, pd.DataFrame | None] | None = None,
          temp: tp.Temperature | None = None) -> str:
    now = now or datetime.now(JST)
    chart_dir = out_dir / "charts"
    sections, missing = [], []
    indexes = {}
    if market is not None:
        indexes = {"グロース250": market.series.get(mk.GROWTH_SYMBOL), "TOPIX": market.series.get(mk.TOPIX_SYMBOL)}
    margin_dates, price_dates = [], []
    grav = en.gravity(market.series) if market is not None and market.series else None

    for code, name in watchlist:
        h = hist_all[hist_all["code"] == code].sort_values("date")
        if h.empty:
            missing.append(f"{name}（{code}）")
            continue
        px = prices.get(code)
        s = summarize(h, px, indexes)
        margin_dates.append(s["date"])
        if s["close_date"] is not None:
            price_dates.append(s["close_date"])
        dist = s["dist"]
        chart(code, name, h, px, s, chart_dir / f"{code}.png")

        price = "" if _missing(s["close"]) else f"　{s['close']:,.0f}円"
        fund = None
        q = (funds or {}).get(code)
        if q is not None and not q.empty:
            fund = fd.evaluate(q, s["close"])
        over = oh.check(px, pd.Timestamp(now.date()), None, None)  # ロックアップ解除の目安だけ使う
        theme_rows = []
        if market is not None:
            for key in (themes or {}).get(code, []):
                found = mk.find_theme(market, key)
                if found:
                    theme_rows.append((found[0], found[2], found[3]))
        sector = en.sector_flow(theme_rows)
        value = en.valuation(fund.per, fund.pbr) if fund is not None else ""
        today = pd.Timestamp(now.date())
        mats = en.materials(code, today, exclude=en.OVERHANG_KINDS)
        heavy_rows, heavy_short = _heavy(code, today, fund, over)
        foot = en.footprints(px, h, indexes.get("グロース250"), code, today)
        sections.append(
            f"### {name}（{code}）{price}\n\n"
            f"> {comment(s, grav, sector, value, heavy_short, foot)}\n\n"
            f"![{name}のグラフ](charts/{code}.png)\n\n"
            f"{_analysis_block(s, h, over, grav, theme_rows, fund, value, mats, heavy_rows, foot)}{_fund_short(fund)}"
        )

    latest = max(margin_dates) if margin_dates else None
    price_day = max(price_dates) if price_dates else None
    meta = []
    if latest is not None:
        meta.append(f"信用残 **{latest:%-m/%-d} 申込み分**")
    if price_day is not None:
        meta.append(f"株価 {price_day:%-m/%-d} 終値")
    meta.append(f"更新 {now:%-m/%-d %H:%M}")

    parts = [
        "# 信用残ウォッチ\n",
        "> " + " ・ ".join(meta) + "\n",
        market_section(market, chart_dir, temp),
    ]
    if missing:
        parts.append("データなし（コードの誤り、または信用取引の対象外）: " + "、".join(missing) + "\n")
    parts += ["## 銘柄\n", "\n\n---\n\n".join(sections), "", _NOTES]
    return "\n".join(parts)


_NOTES = """---

<details><summary><b>各項目の見方</b></summary>

- **考え方**: 株価を動かす一番大きな力は地合いの引力。テクニカルは過去の流れを見るためのもので、
  ターニングポイント（エネルギーが発揮される位置）と、需給で下方向の圧力が助長されているかを確かめるために使う。
- **ひとこと**: 引力・セクター・表面の評価・上値の重し・下方向の圧力・次の転換点・反発をまとめた自動のコメント。
- **温度感**: 市場全体の楽観・悲観のものさし。信用評価損益率（市場全体、推定）は、JPX の全銘柄の信用残のうち
  買残の金額が大きい上位500銘柄について「買残の株数 × 申込日の終値」と「買残の金額（建値）」を比べたもの
  （証券会社や日経の評価損益率とは計算元が違う）。-20%前後で底、-3%前後で天井がよく言われる目安。
  日経平均は25日線乖離（±5%）・RSI（30/70）・ボリンジャーバンド（±2σ）で過熱／売られすぎを見る。
  ものさしの過熱・売られすぎを足して「冷えている（悲観）／普通／熱い（楽観）」を出す。
- **引力**: グロース250・日経平均（25日線の上か下か、5日の騰落±2%）、前夜の S&P500・NASDAQ（±0.5%）、
  VIX（20未満は上、25以上は下）を足して、+2以上なら上向き、-2以下なら下向き。VIX が25以上の山から15%下がると
  「ピークアウト（底の候補）」。
- **セクター**: watchlist.txt で設定した関連テーマ（米国のテーマETF）の前日比と5日比。5日比の平均が±2%で追い風／向かい風。
- **表面の評価**: PER・PBR の水準だけで見た目安（妥当な株価はファンダを見て自分で判断する）。
  PER は 10倍未満=割安・20倍未満=普通・40倍未満=やや割高・それ以上=割高。赤字で PER が付かなければ
  PBR（1倍未満=割安・3倍未満=普通・8倍未満=やや割高・それ以上=割高）で見る。
- **上値の重し**: IR 材料ランキング（trade リポジトリに残してあり、このリポジトリには記録がないので今は出ない）が保存した開示のうち、MSワラント・新株予約権の大量行使・公募増資・売出し・
  第三者割当（新株予約権・転換社債を含む）・大株主の異動（直近1年）と、株数の増加（希薄化）、IPO後のロックアップ解除。
  開示の記録は 2026/11/20 から始まっているので、それより前の材料は手動で確認する。
- **新規材料**: IR 材料ランキング（ir/。trade リポジトリに残してあり、このリポジトリには記録がないので今は出ない）が保存した、直近30日の適時開示（決算短信と、上値の重しに出すものは除く）。かっこ内は材料度。
- **下方向の圧力**: 直近5営業日の株価・信用買残・出来高（その前20日の平均との比）のバランス。
  下げ＋買残増＝助長（強い）、下げ＋買残減＋出来高増＝解放中（投げ）、下げ＋買残減＝弱まる（売り枯れに近い）、
  下げ＋買残横ばい＋閑散＝くすぶる（未精算のじわ下げ）、上げ＋買残増＝上値に重し、上げ＋買残減＝軽い。
- **本尊の気配**: 本尊（主力）の売買は直接見えないので、動いたときに残りやすい足跡を7つ拾う（どれも目安）。
  出来高＝直近20日で上げた日の平均出来高が下げた日の1.5倍以上、OBV＝株価は横ばい・下げなのに出来高の累積が上向き、
  逆行＝10日でグロース250より5%以上強い／逆行陽線2回以上、下値＝直近20日の安値の±1.5%で3回以上止まる、
  売残＝信用売残が20%以上増え、売残が買残の5%以上ある（踏み上げの燃料）、買い集め＝大株主の買い集めの開示（半年）、
  値持ち＝出来高3倍以上で5%以上上げた日の安値を割っていない。3つ以上で「あり」、2つで「少し」。
  手じまい＝直近5日の「出来高2倍以上で長い上ヒゲ」「出来高を伴う直近20日の安値割れ」。
- **反発サイン**: 1日ごとの反発。◎強い反発（前日高値超え、または陽線+3%以上で高値圏引け）・○反発（前日比プラス）・
  △下げ渋り（前日比マイナスだが値幅の上半分）・×反発なし。「逆」はグロース250が-1%以下の日の陽線（逆行陽線）。
- **転換点**: エネルギーが発揮されやすい位置の候補（以下の「抵抗・支え」の根拠から探す）。

- **信用買い方の平均建値**: JPX が公表する買残の金額 ÷ 株数。信用買いしている人たちの平均の買値の目安。
- **買い方損益**: 株価 ÷ 平均建値 − 1。大きなマイナスは、含み損を抱えた買い方が多く、投げ売りや戻り売りが出やすい目安。
- **価格帯別の保有者・含み益の保有者・平均コスト**: 信用に限らず、現物も含めた保有者全体の推定。過去2年の株価と出来高から、
  「毎日、出来高の分だけ保有者が入れ替わり、その日の値幅で買った」とみなして計算したモデル値（moomoo などのコスト分布と同じ考え方）。
  グラフの右側で、緑は含み益、赤は含み損の保有者。保有者が厚い価格帯は、上にあれば戻り売りが出やすい抵抗帯、下にあれば支えの目安。
- **地合い**: VIX・米国（S&P500・NASDAQ・ダウ・SOX・先物・小型株）・米10年金利・日経平均（週足は13週線）・日経先物・
  グロース250・TOPIX・原油・ドル円・ビットコインと、米国のセクター別・テーマ別（サイバーセキュリティ・量子・宇宙など
  35テーマ。各テーマのETFの値）の騰落。株価指数は前日比±0.5%、金利は±0.05pt、
  VIXは20未満=落ち着き・20以上=警戒・30以上=恐怖（直近5日の最高値から15%下がったら「ピークアウト＝底の候補」）。
  原油は±3%、ドル円は±1%、ビットコインは±5%で「注意」。グロース250とTOPIXは連動ETF（2516・1306）の値。
- **信用買残の上場株数比**: JPX が公表する「上場比」。10%以上は需給が重い目安。
- **ファンダ**: Yahoo Finance の四半期決算から、PER（直近4四半期の純利益）・PBR・時価総額を最新の株価で計算し、
  直近四半期を前年同期と比べて評価する（増収増益・増収減益・営業黒字転換など。営業利益を重視）。
  懸念として出すもの: 営業赤字、売上・営業利益 前年同期比 -20%以下、増収減益、営業利益率 3pt以上の悪化（両期とも黒字のとき）、
  株数が1年で3%以上増加（希薄化）、債務超過・有利子負債が自己資本より多い、赤字で現金が8四半期分未満、
  PER 50倍以上・PBR 5倍以上、最新の決算が未反映の可能性。小型株は値の誤りがあり得るので決算短信で確認する。
- **ロックアップ**: IPO後1年以内の銘柄で、上場日から90日後・180日後の解除の目安。解除条件は目論見書で確認する。
- **米国の関連テーマ**: watchlist.txt で銘柄名の後ろに [ ] で書いたテーマ（例: [サイバーセキュリティ, クラウド]）の
  前夜の米国での騰落。テーマ名は地合いのテーマ・業種別の表の名前か、ETF の記号。


- **支えになりやすい位置**: 株価の下にある、買いが入りやすい価格を近い順に3つ（グラフの青緑の線と「下①②③」）。
  根拠は「含み益の保有者が厚い価格帯」「株価より下にある平均建値・平均コスト」「25日・75日移動平均」
  「直近半年で反発した安値（その後10営業日以内に5%以上戻したもの。例: 8/7→+35% は8/7の安値から35%戻した）」
  「埋まっていない上昇の窓」「フィボナッチ50%押し」「ボリンジャー -2σ」「EMA20」「追証ゾーン（直近高値の-10〜20%）」。
  ★の付け方は抵抗と同じ。同じ価格帯で2回以上反発していれば、根拠2つ分と数える。
- **抵抗になりやすい位置**: 株価の上にある、戻り売りが出やすい価格を近い順に3つ（グラフの橙色の線と「上①②③」）。
  根拠は「含み損の保有者が厚い価格帯」「信用買い方の平均建値」「保有者全体の平均コスト」「25日・75日移動平均」
  「直近60営業日で跳ね返された高値（その後10営業日以内に5%以上下げたもの）」「埋まっていない下落の窓」。株価の2%以内で重なる根拠は1つにまとめ、
  根拠が重なるほど強い。表の根拠は短い名前（厚い帯=保有者が厚い価格帯で、範囲とそこに集まる保有者の割合を書く。
  価格はその帯の中で一番保有者が多い価格。グラフでは帯の範囲を薄く塗る。信用建値=信用買い方の平均建値、
  反発安値×2=2回反発した安値、窓=埋まっていない窓）。
- **回転の効き具合**: 出来高のうち何割を「保有者の入れ替わり」とみなしたか（1 = 出来高どおり、0.5 = 半分）。
  出来高には同じ日に買って売るデイトレの分も入るため、銘柄ごとに 0.1〜1.5 の中から、モデルの平均コストが
  JPX の実データ（信用買い方の平均建値）に一番近くなる値を自動で選んでいる（差がわずかなら 1）。
- **買残÷出来高**: 買残が直近20日の平均出来高の何日分か。大きいほど、返済の売りをこなすのに時間がかかる。
- 出典: 信用残は JPX「銘柄別信用取引残高」（毎営業日16時ごろ公表、前営業日の申込み分）。株価・出来高は Yahoo Finance（遅延データ）。

</details>
"""
