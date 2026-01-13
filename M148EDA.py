from __future__ import annotations

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = r"C:\Users\samvi\Downloads\dat_train1.csv"
OUTPUT_DIR = r"C:\Users\samvi\Downloads\m148_eda_outputs"
CHUNKSIZE = 2_000_000

TIME_COL = "event_timestamp"
EVENT_COL = "event_name"

ORDER_EVENT = "order_shipped"
ACCOUNT_START_EVENT = "account_activitation"


def detect_sep(path: str) -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        sample = f.read(20000)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def add_counts(base: pd.Series, inc: pd.Series) -> pd.Series:
    if base is None:
        return inc
    return base.add(inc, fill_value=0)


def month_name_ordered(idx: pd.Index) -> pd.Categorical:
    names = pd.Index(idx).map(lambda m: pd.Timestamp(2000, int(m), 1).strftime("%b"))
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pd.Categorical(names, categories=order, ordered=True)


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def plot_time_series(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str) -> None:
    s = series.sort_index()
    plt.figure()
    plt.plot(s.index, s.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bar(categories, values, title: str, xlabel: str, ylabel: str, outpath: str) -> None:
    plt.figure()
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    sep = detect_sep(DATA_PATH)
    usecols = [TIME_COL, EVENT_COL]
    dtype = {EVENT_COL: "string", TIME_COL: "string"}

    order_daily = None
    order_monthly = None
    order_weekly = None
    order_month_of_year = None
    order_season = None

    acct_daily = None
    acct_monthly = None
    acct_weekly = None
    acct_month_of_year = None
    acct_season = None

    reader = pd.read_csv(DATA_PATH, sep=sep, usecols=usecols, dtype=dtype, chunksize=CHUNKSIZE, low_memory=False)

    for chunk in reader:
        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=[TIME_COL, EVENT_COL])
        if chunk.empty:
            continue

        ts = chunk[TIME_COL]
        ev = chunk[EVENT_COL].astype("string")

        is_order = ev == ORDER_EVENT
        if is_order.any():
            t = ts[is_order]
            d = t.dt.floor("D").value_counts()
            w = t.dt.to_period("W").astype(str).value_counts()
            m = t.dt.to_period("M").astype(str).value_counts()
            moy = t.dt.month.value_counts()
            seas = t.dt.month.map(season_from_month).value_counts()
            order_daily = add_counts(order_daily, d)
            order_weekly = add_counts(order_weekly, w)
            order_monthly = add_counts(order_monthly, m)
            order_month_of_year = add_counts(order_month_of_year, moy)
            order_season = add_counts(order_season, seas)

        is_acct = ev == ACCOUNT_START_EVENT
        if is_acct.any():
            t = ts[is_acct]
            d = t.dt.floor("D").value_counts()
            w = t.dt.to_period("W").astype(str).value_counts()
            m = t.dt.to_period("M").astype(str).value_counts()
            moy = t.dt.month.value_counts()
            seas = t.dt.month.map(season_from_month).value_counts()
            acct_daily = add_counts(acct_daily, d)
            acct_weekly = add_counts(acct_weekly, w)
            acct_monthly = add_counts(acct_monthly, m)
            acct_month_of_year = add_counts(acct_month_of_year, moy)
            acct_season = add_counts(acct_season, seas)

    if order_daily is not None:
        order_daily = order_daily.sort_index()
        plot_time_series(
            order_daily,
            f"Orders per day ({ORDER_EVENT})",
            "Date",
            "Count",
            os.path.join(OUTPUT_DIR, "orders_daily.png"),
        )

    if order_monthly is not None:
        om = order_monthly.sort_index()
        om.index = pd.to_datetime(om.index + "-01", utc=True)
        plot_time_series(
            om,
            f"Orders per month ({ORDER_EVENT})",
            "Month",
            "Count",
            os.path.join(OUTPUT_DIR, "orders_monthly.png"),
        )

    if acct_daily is not None:
        acct_daily = acct_daily.sort_index()
        plot_time_series(
            acct_daily,
            f"Account starts per day ({ACCOUNT_START_EVENT})",
            "Date",
            "Count",
            os.path.join(OUTPUT_DIR, "account_starts_daily.png"),
        )

    if acct_weekly is not None:
        aw = acct_weekly.sort_index()
        x = np.arange(len(aw))
        plt.figure()
        plt.plot(x, aw.values)
        plt.title(f"Account starts per week ({ACCOUNT_START_EVENT})")
        plt.xlabel("Week index (chronological)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "account_starts_weekly.png"), dpi=200)
        plt.close()

    if acct_monthly is not None:
        am = acct_monthly.sort_index()
        am.index = pd.to_datetime(am.index + "-01", utc=True)
        plot_time_series(
            am,
            f"Account starts per month ({ACCOUNT_START_EVENT})",
            "Month",
            "Count",
            os.path.join(OUTPUT_DIR, "account_starts_monthly.png"),
        )

    if order_month_of_year is not None:
        moy = order_month_of_year.reindex(range(1, 13)).fillna(0)
        cats = month_name_ordered(moy.index)
        dfm = pd.DataFrame({"month": cats, "count": moy.values}).sort_values("month")
        plot_bar(
            dfm["month"].astype(str).values,
            dfm["count"].values,
            f"Orders by month-of-year (all years combined)",
            "Month",
            "Count",
            os.path.join(OUTPUT_DIR, "orders_by_month_of_year.png"),
        )

    if acct_month_of_year is not None:
        moy = acct_month_of_year.reindex(range(1, 13)).fillna(0)
        cats = month_name_ordered(moy.index)
        dfm = pd.DataFrame({"month": cats, "count": moy.values}).sort_values("month")
        plot_bar(
            dfm["month"].astype(str).values,
            dfm["count"].values,
            f"Account starts by month-of-year (all years combined)",
            "Month",
            "Count",
            os.path.join(OUTPUT_DIR, "account_starts_by_month_of_year.png"),
        )

    if order_season is not None:
        seas_order = ["DJF", "MAM", "JJA", "SON"]
        s = order_season.reindex(seas_order).fillna(0)
        plot_bar(
            s.index.astype(str).values,
            s.values,
            "Orders by season (all years combined)",
            "Season",
            "Count",
            os.path.join(OUTPUT_DIR, "orders_by_season.png"),
        )

    if acct_season is not None:
        seas_order = ["DJF", "MAM", "JJA", "SON"]
        s = acct_season.reindex(seas_order).fillna(0)
        plot_bar(
            s.index.astype(str).values,
            s.values,
            "Account starts by season (all years combined)",
            "Season",
            "Count",
            os.path.join(OUTPUT_DIR, "account_starts_by_season.png"),
        )

    summary_rows = []
    def add_summary(name: str, s: pd.Series | None):
        if s is None or s.empty:
            return {
                "metric": name,
                "total_count": 0,
                "mean": np.nan,
                "median": np.nan,
                "max": np.nan,
        }
        s = s.astype(float)
        return {
            "metric": name,
            "total_count": float(s.sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "max": float(s.max()),
    }


    add_summary("orders_daily_total", order_daily)
    add_summary("orders_monthly_total", order_monthly)
    add_summary("account_starts_daily_total", acct_daily)
    add_summary("account_starts_monthly_total", acct_monthly)

    summary = pd.DataFrame([
    add_summary("orders_daily", order_daily),
    add_summary("orders_monthly", order_monthly),
    add_summary("account_starts_daily", acct_daily),
    add_summary("account_starts_monthly", acct_monthly),
    ])    
    summary_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)

    print("Saved outputs to:", OUTPUT_DIR)
    print("Summary stats:", summary_path)
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.endswith(".png"):
            print("Plot:", os.path.join(OUTPUT_DIR, fn))


if __name__ == "__main__":
    main()
