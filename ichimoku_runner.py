# ichimoku_runner.py
# Python 3.11+
# Generuje:
# - email_body.html (HTML treści maila)
# - email_subject.txt (temat maila z liczbą sygnałów)
#
# Wymaga: yfinance pandas numpy requests lxml

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Literal

import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

IndexName = Literal["SP500", "DJIA", "NASDAQ100", "DAX40", "CAC40", "WIG20", "MWIG40"]


@dataclass(frozen=True)
class CrossResult:
    index: IndexName
    ticker: str
    direction: Literal["bullish", "bearish"]


# =============================
# HTTP helper
# =============================
def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def read_html_tables(url: str) -> list[pd.DataFrame]:
    html = fetch_html(url)
    return pd.read_html(StringIO(html))


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    return None


# =============================
# Ticker normalization
# =============================
def to_yahoo_us(t: str) -> str:
    return str(t).strip().replace(".", "-")


def normalize_dax_symbol(raw: str) -> str:
    x = str(raw).strip()
    x = x.replace("–", "-").replace("—", "-").replace(".", "-")
    x = re.sub(r"-(DEU|DE)$", "", x, flags=re.IGNORECASE)
    if "." in x:
        return x
    return f"{x}.DE"


def normalize_cac_symbol(raw: str) -> str:
    x = str(raw).strip().replace("–", "-").replace("—", "-")
    if "." in x:
        return x
    return f"{x}.PA"


def normalize_gpw_to_yahoo(raw: str) -> str:
    x = str(raw).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,10}", x):
        return ""
    return f"{x}.WA"


# =============================
# Index constituents
# =============================
def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = read_html_tables(url)[0]
    col = pick_col(df, ["Symbol"])
    if not col:
        raise RuntimeError("S&P500: nie znaleziono kolumny Symbol.")
    tickers = [to_yahoo_us(t) for t in df[col].astype(str).tolist()]
    return sorted(set([t for t in tickers if t and t.lower() != "nan"]))


def get_djia_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = read_html_tables(url)

    best = None
    best_score = -1
    for t in tables:
        col = pick_col(t, ["Symbol", "Ticker"])
        if not col:
            continue
        score = len(t)
        if 20 <= score <= 60 and score > best_score:
            best = (t, col)
            best_score = score

    if not best:
        raise RuntimeError("DJIA: nie znaleziono tabeli komponentów.")
    df, col = best
    tickers = [to_yahoo_us(x) for x in df[col].astype(str).tolist()]
    return sorted(set([t for t in tickers if t and t.lower() != "nan"]))


def get_nasdaq100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = read_html_tables(url)

    best = None
    best_score = -1
    for t in tables:
        col = pick_col(t, ["Ticker", "Symbol"])
        if not col:
            continue
        score = len(t)
        if score > best_score:
            best = (t, col)
            best_score = score

    if not best or best_score < 80:
        raise RuntimeError("NASDAQ100: nie znaleziono tabeli komponentów.")
    df, col = best
    tickers = [to_yahoo_us(x) for x in df[col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t.lower() != "nan" and " " not in t]
    return sorted(set(tickers))


def get_dax40_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/DAX"
    tables = read_html_tables(url)

    best = None
    best_score = -1
    for t in tables:
        col = pick_col(t, ["Ticker", "Symbol"])
        if not col:
            continue
        score = len(t)
        if 30 <= score <= 60 and score > best_score:
            best = (t, col)
            best_score = score

    if not best:
        raise RuntimeError("DAX40: nie znaleziono tabeli komponentów.")
    df, col = best
    tickers = [normalize_dax_symbol(x) for x in df[col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t.lower() != "nan" and " " not in t]
    return sorted(set(tickers))


def get_cac40_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/CAC_40"
    tables = read_html_tables(url)

    best = None
    best_score = -1
    for t in tables:
        col = pick_col(t, ["Ticker", "Symbol"])
        if not col:
            continue
        score = len(t)
        if 30 <= score <= 60 and score > best_score:
            best = (t, col)
            best_score = score

    if not best:
        raise RuntimeError("CAC40: nie znaleziono tabeli komponentów.")
    df, col = best
    tickers = [normalize_cac_symbol(x) for x in df[col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t.lower() != "nan" and " " not in t]
    return sorted(set(tickers))


def get_stooq_index_tickers(symbol: str) -> list[str]:
    url = f"https://stooq.pl/q/i/?s={symbol}"
    tables = read_html_tables(url)

    best = None
    for t in tables:
        col = pick_col(t, ["Symbol", "Ticker", "Walor"])
        if col:
            best = (t, col)
            break
    if not best:
        t = tables[0]
        col = t.columns[0]
        best = (t, col)

    df, col = best
    raw = df[col].astype(str).tolist()

    tickers = []
    for x in raw:
        base = str(x).strip().split(".")[0].upper()
        if re.fullmatch(r"[A-Z0-9]{1,10}", base):
            tickers.append(base)
    return sorted(set(tickers))


def get_wig20() -> list[str]:
    return [normalize_gpw_to_yahoo(t) for t in get_stooq_index_tickers("wig20") if normalize_gpw_to_yahoo(t)]


def get_mwig40() -> list[str]:
    return [normalize_gpw_to_yahoo(t) for t in get_stooq_index_tickers("mwig40") if normalize_gpw_to_yahoo(t)]


# =============================
# Ichimoku
# =============================
def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["High"]
    low = out["Low"]

    out["tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(26)
    out["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return out


def check_signal(df: pd.DataFrame) -> str | None:
    if len(df) < 90:
        return None

    df = df.dropna(subset=["tenkan", "kijun", "senkou_a", "senkou_b", "Close"])
    if len(df) < 2:
        return None

    t_db = float(df["tenkan"].iloc[-2])
    k_db = float(df["kijun"].iloc[-2])

    t_y = float(df["tenkan"].iloc[-1])
    k_y = float(df["kijun"].iloc[-1])
    close_y = float(df["Close"].iloc[-1])

    sa = float(df["senkou_a"].iloc[-1])
    sb = float(df["senkou_b"].iloc[-1])
    cloud_upper = max(sa, sb)
    cloud_lower = min(sa, sb)

    bullish_cross = (t_db <= k_db) and (t_y > k_y)
    bearish_cross = (t_db >= k_db) and (t_y < k_y)

    if bullish_cross and close_y > cloud_upper:
        return "bullish"
    if bearish_cross and close_y < cloud_lower:
        return "bearish"
    return None


# =============================
# Batch download + retry
# =============================
def split_chunks(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def download_once(batch: list[str], period: str) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=batch,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out

    if not isinstance(data.columns, pd.MultiIndex):
        if {"High", "Low", "Close"}.issubset(data.columns):
            df = data.dropna(subset=["High", "Low", "Close"])
            if not df.empty:
                out[batch[0]] = df
        return out

    available = set(data.columns.get_level_values(0))
    for t in batch:
        if t not in available:
            continue
        df = data[t].copy()
        if df is None or df.empty:
            continue
        if {"High", "Low", "Close"}.issubset(df.columns):
            df = df.dropna(subset=["High", "Low", "Close"])
            if not df.empty:
                out[t] = df
    return out


def batch_download_with_retry(
    tickers: list[str],
    period: str = "1y",
    chunk: int = 60,
    retries: int = 3,
    backoff_s: float = 2.0,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    ok: dict[str, pd.DataFrame] = {}
    remaining = list(dict.fromkeys(tickers))

    for attempt in range(retries + 1):
        if not remaining:
            break

        for part in split_chunks(remaining, chunk):
            got = download_once(part, period)
            ok.update(got)

        remaining = [t for t in remaining if t not in ok]

        if remaining and attempt < retries:
            time.sleep(backoff_s * (attempt + 1))

    return ok, remaining


def screen_index(index: IndexName, tickers: list[str]) -> tuple[list[CrossResult], list[tuple[str, str]]]:
    price_map, failed = batch_download_with_retry(tickers=tickers, period="1y")
    results: list[CrossResult] = []
    for ticker, df in price_map.items():
        try:
            df = add_ichimoku(df)
            signal = check_signal(df)
            if signal:
                results.append(CrossResult(index=index, ticker=ticker, direction=signal))  # type: ignore[arg-type]
        except Exception:
            continue
    return results, [(index, t) for t in failed]


# =============================
# HTML Email formatting
# =============================
def escape_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def html_table(rows: list[tuple[str, str]], title: str) -> str:
    # rows: [(Index, "ticker1, ticker2,..."), ...]
    if not rows:
        return f"""
        <h3 style="margin:16px 0 8px 0;">{escape_html(title)}</h3>
        <p style="margin:0;">Brak</p>
        """

    tr = []
    for idx, tickers in rows:
        tr.append(
            f"<tr>"
            f"<td style='padding:8px;border:1px solid #ddd;vertical-align:top;'><b>{escape_html(idx)}</b></td>"
            f"<td style='padding:8px;border:1px solid #ddd;'>{escape_html(tickers)}</td>"
            f"</tr>"
        )
    return f"""
    <h3 style="margin:16px 0 8px 0;">{escape_html(title)}</h3>
    <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:14px;">
      <thead>
        <tr>
          <th style="text-align:left;padding:8px;border:1px solid #ddd;background:#f6f6f6;">Index</th>
          <th style="text-align:left;padding:8px;border:1px solid #ddd;background:#f6f6f6;">Tickers</th>
        </tr>
      </thead>
      <tbody>
        {''.join(tr)}
      </tbody>
    </table>
    """


def build_html_email(df_out: pd.DataFrame, df_failed: pd.DataFrame, run_dt: datetime) -> tuple[str, str]:
    # subject
    n_signals = len(df_out)
    date_str = run_dt.strftime("%Y-%m-%d")
    subject = f"Ichimoku — {n_signals} sygnałów — {date_str}"

    # group by direction & index
    bullish_rows: list[tuple[str, str]] = []
    bearish_rows: list[tuple[str, str]] = []

    if not df_out.empty:
        for idx in sorted(df_out["index"].unique()):
            sub = df_out[df_out["index"] == idx]
            b = sub[sub["direction"] == "bullish"]["ticker"].tolist()
            s = sub[sub["direction"] == "bearish"]["ticker"].tolist()
            if b:
                bullish_rows.append((idx, ", ".join(b)))
            if s:
                bearish_rows.append((idx, ", ".join(s)))

    # optional failed summary
    failed_html = ""
    if not df_failed.empty:
        show = df_failed.head(30)
        grouped = show.groupby("index")["ticker"].apply(list).to_dict()
        li = []
        for idx, tks in grouped.items():
            li.append(f"<li><b>{escape_html(idx)}</b>: {escape_html(', '.join(tks))}</li>")
        more = ""
        if len(df_failed) > 30:
            more = f"<p style='margin:8px 0 0 0;color:#666;'>… oraz {len(df_failed) - 30} kolejnych</p>"

        failed_html = f"""
        <h3 style="margin:20px 0 8px 0;">Tickery bez danych/timeout ({len(df_failed)})</h3>
        <ul style="margin:0 0 0 18px;padding:0;font-family:Arial,sans-serif;font-size:14px;">
          {''.join(li)}
        </ul>
        {more}
        """

    run_str = run_dt.strftime("%Y-%m-%d %H:%M:%S")

    body = f"""<!doctype html>
<html>
  <body style="font-family:Arial,sans-serif;">
    <h2 style="margin:0 0 6px 0;">Ichimoku screener</h2>
    <p style="margin:0 0 14px 0;color:#555;">
      Tenkan/Kijun cross (wczoraj) + filtr ceny względem chmury (dziś).<br/>
      Run: <b>{escape_html(run_str)}</b> (Europe/Warsaw)
    </p>

    {html_table(bullish_rows, "Bullish (cena nad chmurą)")}
    {html_table(bearish_rows, "Bearish (cena pod chmurą)")}

    {failed_html}

    <hr style="border:none;border-top:1px solid #eee;margin:18px 0;">
    <p style="margin:0;color:#777;font-size:12px;">
      Wiadomość wygenerowana automatycznie przez GitHub Actions.
    </p>
  </body>
</html>
"""
    return subject, body


# =============================
# MAIN
# =============================
def main():
    run_dt = datetime.now()
    print("Pobieram listy spółek...")

    universe: dict[IndexName, list[str]] = {
        "SP500": get_sp500_tickers(),
        "DJIA": get_djia_tickers(),
        "NASDAQ100": get_nasdaq100_tickers(),
        "DAX40": get_dax40_tickers(),
        "CAC40": get_cac40_tickers(),
        "WIG20": get_wig20(),
        "MWIG40": get_mwig40(),
    }

    all_results: list[CrossResult] = []
    all_failed: list[tuple[str, str]] = []

    for idx, tickers in universe.items():
        print(f"Skanuję {idx} ({len(tickers)} spółek)...")
        res, failed_pairs = screen_index(idx, tickers)
        all_results.extend(res)
        all_failed.extend(failed_pairs)
        if failed_pairs:
            print(f"  -> nie pobrano danych dla: {len(failed_pairs)} tickerów (brak danych/timeout)")

    if not all_results:
        df_out = pd.DataFrame(columns=["index", "ticker", "direction"])
        print("\nBrak sygnałów.")
    else:
        df_out = pd.DataFrame([r.__dict__ for r in all_results])[["index", "ticker", "direction"]]
        df_out = df_out.sort_values(["direction", "index", "ticker"])
        print("\nSYGNAŁY:")
        print(df_out.to_string(index=False))

    if all_failed:
        df_failed = pd.DataFrame(all_failed, columns=["index", "ticker"])
    else:
        df_failed = pd.DataFrame(columns=["index", "ticker"])

    subject, html = build_html_email(df_out, df_failed, run_dt)

    with open("email_subject.txt", "w", encoding="utf-8") as f:
        f.write(subject.strip())

    with open("email_body.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Zapisano: email_subject.txt i email_body.html")
    print("Koniec skanu.")


if __name__ == "__main__":
    main()
