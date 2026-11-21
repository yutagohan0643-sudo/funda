"""CSV / Markdown レポート出力."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .config import MODE_LABELS

import pandas as pd


def _pct(x):
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def _num(x, fmt="{:.1f}"):
    return "-" if pd.isna(x) else fmt.format(x)


def write(df: pd.DataFrame, out_dir: Path, top: int, cfg) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["コード", "銘柄名", "市場", "業種", "score", "regularMarketPrice",
            "marketCap", "trailingPE", "cash_adj_per", "priceToBook",
            "ev_ebitda", "returnOnEquity", "operatingMargins", "revenueGrowth",
            "earningsGrowth", "net_cash_ratio", "dividendYield", "analysts",
            "heldPercentInstitutions", "heldPercentInsiders", "turnover",
            "fiftyTwoWeekChangePercent", "s_undiscovered", "s_value",
            "s_quality", "s_growth", "one_off_flag", "per_calc", "data_flag", "website"]
    df[[c for c in cols if c in df.columns]].to_csv(
        out_dir / "candidates.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"# {'ファンダメンタルズ' if cfg.mode == 'fundamental' else '未発掘銘柄'}スクリーニング ({date.today():%Y-%m-%d})", "",
        "## 条件", "",
        f"- 時価総額 {cfg.min_mcap/1e8:.0f}億〜{cfg.max_mcap/1e8:.0f}億円 / "
        f"PER ≤ {cfg.max_per} / PBR ≤ {cfg.max_pbr} / ROE ≥ {cfg.min_roe*100:.0f}%",
        (f"- アナリストカバレッジ ≤ {cfg.max_analysts}人 / " if cfg.max_analysts is not None else "- ")
        + f"平均売買代金 ≥ {cfg.min_turnover/1e6:.0f}百万円/日 / 金融業除外",
        f"- 通過銘柄数: {len(df)}", "",
        f"スコア = {MODE_LABELS[cfg.mode]} (各指標の順位ベース, 0-100)", "",
        "- 割安: 現金調整PER・PBR・EV/EBITDA・純現金/時価総額",
        "- 質: ROE・ROA・営業利益率・流動比率・純現金・インサイダー保有",
        "- 成長: 売上成長率・利益成長率", "",
        f"## 上位 {min(top, len(df))} 銘柄", "",
        "| # | コード | 銘柄名 | 市場 | 業種 | スコア | 時価総額(億) | PER | 現金調整PER | PBR | ROE | 営業利益率 | 売上成長 | 純現金/時価 | 配当 | アナリスト | 機関保有 | 注意 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(df.head(top).itertuples(index=False), 1):
        r = r._asdict()
        dy = r.get("dividendYield")
        lines.append(
            f"| {i} | {r['コード']} | {r['銘柄名']} | {r['市場']} | {r['業種']} | "
            f"{r['score']:.1f} | {r['marketCap']/1e8:,.0f} | {_num(r['trailingPE'])} | "
            f"{_num(r['cash_adj_per'])} | {_num(r['priceToBook'], '{:.2f}')} | "
            f"{_pct(r['returnOnEquity'])} | {_pct(r['operatingMargins'])} | "
            f"{_pct(r['revenueGrowth'])} | {_pct(r['net_cash_ratio'])} | "
            f"{_num(dy, '{:.2f}%')} | {r['analysts']} | {_pct(r['heldPercentInstitutions'])} | {' '.join(f for f, on in (('⚠特益?', r['one_off_flag']), ('⚠データ', r['data_flag'])) if on)} |"
        )
    lines += ["", "> データ: JPX 上場銘柄一覧, Yahoo Finance。投資判断は一次情報"
              "(決算短信・有報)で必ず確認してください。投資助言ではありません。",
              "> ⚠特益?: 純利益率が営業利益率を大きく上回り、特別利益で PER が低く見えている可能性 (スコア15%減点)。",
              "> ⚠データ: 時価総額÷純利益で再計算した PER と表示 PER が乖離 (スコア30%減点、要一次情報確認)。",
              "> Yahoo の日本小型株データはアナリスト数・機関保有比率が欠損(0扱い)のことが多い点に注意。", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
