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
from bs4 import BeautifulSoup

# wycisz spam z yfinance
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
# Normalizacja tickerów
# =============================
def to_yahoo_us(t: str) -> str:
    return str(t).strip().replace(".", "-")  # BRK.B -> BRK-B


def normalize_dax_symbol(raw: str) -> str:
    x = str(raw).strip()
    x = x.replace("–", "-").replace("—", "-").replace(".", "-")
    x = re.sub(r"-(DEU|DE)$", "", x, flags=re.IGNORECASE)  # usuń końcówki -DE/-DEU
    if "." in x:
        return x
    return f"{x}.DE"


def normalize_cac_symbol(raw: str) -> str:
    x = str(raw).strip().replace("–", "-").replace("—", "-")
    if "." in x:
        return x
    return f"{x}.PA"


def normalize_gpw_to_yahoo(raw: str) -> str:
    """
    Stooq zwraca tickery GPW w stylu: 11B, PKO, PZU, XTB...
    Dla Yahoo: .WA
    """
    x = str(raw).strip().upper()
    # tylko alfanumeryczne (GPW tickery)
    if not re.fullmatch(r"[A-Z0-9]{1,10}", x):
        return ""
    return f"{x}.WA"


# =============================
# Składy indeksów
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


# --- Stooq: skład indeksu ---
def get_stooq_index_tickers(symbol: str) -> list[str]:
    """
    Stooq:
      wig20  -> https://stooq.pl/q/i/?s=wig20
      mwig40 -> https://stooq.pl/q/i/?s=mwig40
    Zwraca listę tickerów (bez .WA), potem mapujemy na Yahoo: .WA
    """
    url = f"https://stooq.pl/q/i/?s={symbol}"
    tables = read_html_tables(url)

    # zwykle jest jedna tabela; szukamy kolumny "Symbol"
    best = None
    for t in tables:
        col = pick_col(t, ["Symbol", "Ticker", "Walor"])
        if col:
            best = (t, col)
            break

    if not best:
        # fallback: spróbuj wyciągnąć z pierwszej tabeli pierwszą kolumnę
        t = tables[0]
        col = t.columns[0]
        best = (t, col)

    df, col = best
    raw = df[col].astype(str).tolist()

    tickers = []
    for x in raw:
        # czasem mogą być wpisy typu "11B" lub "11B.PL" – bierzemy część przed kropką
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
    return [xs[i:i+n] for i in range(0, len(xs), n)]


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

    # 1 ticker -> flat DF
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


def screen_index(index: IndexName, tickers: list[str]) -> tuple[list[CrossResult], list[str]]:
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

    return results, failed


# =============================
# MAIN
# =============================
def main():
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

    expected = {"DJIA": 30, "DAX40": 40, "CAC40": 40, "WIG20": 20, "MWIG40": 40}
    for idx, lst in universe.items():
        if idx in expected and len(lst) < expected[idx]:
            print(f"[WARN] {idx}: pobrano {len(lst)} zamiast ~{expected[idx]} (sprawdź źródło listy).")

    all_results: list[CrossResult] = []
    all_failed: list[tuple[str, str]] = []

    for idx, tickers in universe.items():
        print(f"Skanuję {idx} ({len(tickers)} spółek)...")
        res, failed = screen_index(idx, tickers)
        all_results.extend(res)
        all_failed.extend([(idx, t) for t in failed])
        if failed:
            print(f"  -> nie pobrano danych dla: {len(failed)} tickerów (brak danych/timeout)")

    if not all_results:
        print("\nBrak sygnałów.")
        df_out = pd.DataFrame(columns=["index", "ticker", "direction"])
    else:
        df_out = pd.DataFrame([r.__dict__ for r in all_results])[["index", "ticker", "direction"]]
        df_out = df_out.sort_values(["index", "direction", "ticker"])
        print("\nSYGNAŁY:")
        print(df_out.to_string(index=False))

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    signals_file = f"ichimoku_signals_{ts}.csv"
    df_out.to_csv(signals_file, index=False, encoding="utf-8")
    print(f"\nZapisano plik: {signals_file}")

    if all_failed:
        df_failed = pd.DataFrame(all_failed, columns=["index", "ticker"])
        failed_file = f"ichimoku_failed_{ts}.csv"
        df_failed.to_csv(failed_file, index=False, encoding="utf-8")
        print(f"Zapisano plik: {failed_file}")


if __name__ == "__main__":
    main()


df_out.to_csv("signals.csv", index=False, encoding="utf-8")

if all_failed:
    df_failed = pd.DataFrame(all_failed, columns=["index", "ticker"])
    df_failed.to_csv("failed.csv", index=False, encoding="utf-8")
else:
    #żeby plik istniał zawsze (opcjonalnie)
    pd.DataFrame(columns=["index", "ticker"]).to_csv("failed.csv", index=False, encoding="utf-8")
