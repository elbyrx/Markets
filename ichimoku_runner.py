# ichimoku_runner.py
# Python 3.11+
# Generuje:
# - signals.csv / failed.csv (opcjonalnie zostawiamy jako artefakty/debug)
# - email_body.txt (treść maila — TO wykorzysta workflow)
#
# Wymaga: yfinance pandas numpy requests beautifulsoup4 lxml

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
    return str(t).strip().replace(".", "-")  # BRK.B -> BRK-B


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
    # symbol: "wig20", "mwig40"
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

    # single ticker -> flat
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
    failed_pairs = [(index, t) for t in failed]
    return results, failed_pairs


# =============================
# Email body formatting
# =============================
def format_email_body(df_out: pd.DataFrame, df_failed: pd.DataFrame, run_dt: datetime) -> str:
    ts = run_dt.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append(f"Ichimoku screener (Tenkan/Kijun cross + cena vs chmura)")
    lines.append(f"Run: {ts} (Europe/Warsaw)")
    lines.append("")

    if df_out.empty:
        lines.append("Brak sygnałów na dziś.")
    else:
        lines.append(f"Sygnały: {len(df_out)}")
        lines.append("")
        # Grupowanie: index -> direction -> tickers
        for idx in sorted(df_out["index"].unique()):
            sub = df_out[df_out["index"] == idx]
            lines.append(f"[{idx}]")
            for direction in ["bullish", "bearish"]:
                sub2 = sub[sub["direction"] == direction]
                if sub2.empty:
                    continue
                tickers = ", ".join(sub2["ticker"].tolist())
                lines.append(f"  {direction}: {tickers}")
            lines.append("")

    # Failed (opcjonalnie, krótko)
    if not df_failed.empty:
        lines.append(f"Tickery bez danych/timeout: {len(df_failed)}")
        # pokaż max 30, żeby nie spamować
        show = df_failed.head(30)
        grouped = show.groupby("index")["ticker"].apply(list).to_dict()
        for idx, tks in grouped.items():
            lines.append(f"  {idx}: {', '.join(tks)}")
        if len(df_failed) > 30:
            lines.append(f"  ... oraz {len(df_failed) - 30} kolejnych")
        lines.append("")

    lines.append("—")
    lines.append("Wiadomość wygenerowana automatycznie przez GitHub Actions.")
    return "\n".join(lines)


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

    # df_out / df_failed zawsze zdefiniowane
    if not all_results:
        print("\nBrak sygnałów.")
        df_out = pd.DataFrame(columns=["index", "ticker", "direction"])
    else:
        df_out = pd.DataFrame([r.__dict__ for r in all_results])[["index", "ticker", "direction"]]
        df_out = df_out.sort_values(["index", "direction", "ticker"])
        print("\nSYGNAŁY:")
        print(df_out.to_string(index=False))

    if all_failed:
        df_failed = pd.DataFrame(all_failed, columns=["index", "ticker"])
    else:
        df_failed = pd.DataFrame(columns=["index", "ticker"])

    # (opcjonalnie) zostawiamy CSV jako debug/artefakty
    df_out.to_csv("signals.csv", index=False, encoding="utf-8")
    df_failed.to_csv("failed.csv", index=False, encoding="utf-8")

    # Treść maila do pliku
    body = format_email_body(df_out, df_failed, run_dt)
    with open("email_body.txt", "w", encoding="utf-8") as f:
        f.write(body)

    print("Zapisano: email_body.txt (treść maila)")
    print("Koniec skanu.")


if __name__ == "__main__":
    main()
