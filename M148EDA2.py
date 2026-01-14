from __future__ import annotations

import os
import csv
import glob
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

## Parts of this code were generated using AI (ChatGPT).

DATA_PATH = r"C:\Users\samvi\Downloads\dat_train1.csv"
OUTPUT_PDF = r"C:\Users\samvi\Downloads\journey_eda_report_full_fast_days.pdf"
WORK_DIR = r"C:\Users\samvi\Downloads\m148_eda_work_full_fast"

CHUNKSIZE = 1_500_000
N_PARTITIONS = 256

TIME_COL = "event_timestamp"
CUST_COL = "customer_id"
ACCT_COL = "account_id"
ACTION_ID_COL = "ed_id"
ACTION_NAME_COL = "event_name"

SUCCESS_EVENT = "order_shipped"
INACTIVITY_DAYS = 60
INACTIVITY = pd.Timedelta(days=INACTIVITY_DAYS)


def detect_sep(path: str) -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as f:
        sample = f.read(20000)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stable_hash_pair(a: pd.Series, b: pd.Series) -> pd.Series:
    key = a.astype("string").fillna("") + "|" + b.astype("string").fillna("")
    return pd.util.hash_pandas_object(key, index=False)


def df_to_rl_table(df: pd.DataFrame, max_rows: int = 35, floatfmt: str = "{:.4g}") -> Table:
    view = df.head(max_rows).copy()

    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    data = [list(view.columns)] + [[fmt(v) for v in row] for row in view.to_numpy()]
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    return tbl


def summarize_numeric(series: pd.Series, name: str) -> pd.DataFrame:
    s = series.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["metric", name])
    qs = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
    rows = {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "min": float(s.min()),
        **{f"p{int(k*100):02d}": float(v) for k, v in qs.items()},
        "max": float(s.max()),
    }
    return pd.DataFrame({"metric": list(rows.keys()), name: list(rows.values())})


def save_hist_png(series: pd.Series, title: str, xlabel: str, path: str, bins: int = 80, logy: bool = True):
    plt.figure()
    s = series.dropna()
    plt.hist(s, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_bar_png(labels: pd.Series, values: pd.Series, title: str, xlabel: str, ylabel: str, path: str):
    plt.figure()
    plt.bar(labels.astype(str), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_pdf_report(
    output_pdf: str,
    meta: dict,
    actions_tbl: pd.DataFrame,
    actions_count_stats: pd.DataFrame,
    duration_stats: pd.DataFrame,
    gap_stats: pd.DataFrame,
    outcome_counts: pd.DataFrame,
    png_paths: list[str],
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    story = []
    story.append(Paragraph("Customer Journey EDA Report (Full Dataset)", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    meta_lines = [
        f"<b>Data file:</b> {meta['data_path']}",
        f"<b>Delimiter:</b> {repr(meta['sep'])}",
        f"<b>Events processed:</b> {meta['n_events']:,}",
        f"<b>Journeys:</b> {meta['n_journeys']:,}",
        f"<b>Inactivity threshold:</b> {INACTIVITY_DAYS} days",
        f"<b>Success event:</b> {SUCCESS_EVENT}",
        f"<b>Dataset window:</b> {meta['min_time']} to {meta['max_time']}",
        f"<b>Partitions:</b> {meta['n_partitions']}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Journey outcomes", styles["Heading2"]))
    story.append(Paragraph(
        "Outcomes are labeled as SUCCESS if a journey contains <b>order_shipped</b>, "
        "INACTIVITY if it ends with ≥60 days of inactivity within the observed window, "
        "and CENSORED if the observed dataset ends before we can observe 60 days of inactivity.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 0.12 * inch))
    story.append(df_to_rl_table(outcome_counts))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Journey length and timing", styles["Heading2"]))
    story.append(Paragraph("<b>Actions per journey</b>", styles["Heading3"]))
    story.append(df_to_rl_table(actions_count_stats))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("<b>Journey duration (days)</b>", styles["Heading3"]))
    story.append(df_to_rl_table(duration_stats))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("<b>Inter-action gaps (days)</b>", styles["Heading3"]))
    story.append(df_to_rl_table(gap_stats))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Common actions", styles["Heading2"]))
    story.append(df_to_rl_table(actions_tbl))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Figures", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))

    for p in png_paths:
        story.append(Image(p, width=7.2 * inch, height=4.2 * inch))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


def prepare_partition_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[CUST_COL] = df[CUST_COL].astype("string")
    df[ACCT_COL] = df[ACCT_COL].astype("string")
    df[ACTION_NAME_COL] = df[ACTION_NAME_COL].astype("string")
    df[ACTION_ID_COL] = pd.to_numeric(df[ACTION_ID_COL], errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL, CUST_COL, ACCT_COL])
    df = df.sort_values([CUST_COL, ACCT_COL, TIME_COL]).reset_index(drop=True)
    return df


def build_journeys_and_gaps(df: pd.DataFrame, global_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["prev_time"] = df.groupby([CUST_COL, ACCT_COL])[TIME_COL].shift()
    df["gap"] = df[TIME_COL] - df["prev_time"]
    df["new_journey"] = df["prev_time"].isna() | (df["gap"] > INACTIVITY)
    df["journey_index"] = df.groupby([CUST_COL, ACCT_COL])["new_journey"].cumsum()
    df["journey_id"] = df[CUST_COL].astype(str) + "_" + df[ACCT_COL].astype(str) + "_" + df["journey_index"].astype(str)

    g = df.groupby("journey_id", sort=False)
    journeys = pd.DataFrame({
        "customer_id": g[CUST_COL].first(),
        "account_id": g[ACCT_COL].first(),
        "start_time": g[TIME_COL].min(),
        "end_time": g[TIME_COL].max(),
        "actions_count": g.size(),
        "has_success": g[ACTION_NAME_COL].apply(lambda x: (x == SUCCESS_EVENT).any()),
    })
    journeys["duration_seconds"] = (journeys["end_time"] - journeys["start_time"]).dt.total_seconds()
    journeys["duration_days"] = journeys["duration_seconds"] / (24.0 * 3600.0)
    journeys["is_censored"] = (~journeys["has_success"]) & ((global_end - journeys["end_time"]) < INACTIVITY)
    journeys["outcome"] = np.where(journeys["has_success"], "SUCCESS",
                            np.where(journeys["is_censored"], "CENSORED", "INACTIVITY"))
    journeys = journeys.reset_index()

    gaps_days = (df.groupby("journey_id")[TIME_COL].diff().dt.total_seconds() / (24.0 * 3600.0)).dropna()
    return journeys, gaps_days


def main() -> None:
    ensure_dir(WORK_DIR)
    part_dir = os.path.join(WORK_DIR, "partitions")
    ensure_dir(part_dir)

    sep = detect_sep(DATA_PATH)

    usecols = [CUST_COL, ACCT_COL, ACTION_ID_COL, ACTION_NAME_COL, TIME_COL]
    dtypes = {
        CUST_COL: "string",
        ACCT_COL: "string",
        ACTION_ID_COL: "float64",
        ACTION_NAME_COL: "string",
        TIME_COL: "string",
    }

    try:
        import pyarrow  # noqa: F401
    except Exception:
        raise RuntimeError("Install pyarrow: python -m pip install pyarrow")

    for f in glob.glob(os.path.join(part_dir, "part_*.parquet")):
        os.remove(f)

    n_events_total = 0
    min_time = None
    max_time = None

    reader = pd.read_csv(
        DATA_PATH,
        sep=sep,
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )

    chunk_id = 0
    for chunk in reader:
        chunk = chunk.dropna(subset=[CUST_COL, ACCT_COL, TIME_COL, ACTION_NAME_COL])
        if chunk.empty:
            chunk_id += 1
            continue

        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=[TIME_COL])
        if chunk.empty:
            chunk_id += 1
            continue

        n_events_total += len(chunk)
        cmin = chunk[TIME_COL].min()
        cmax = chunk[TIME_COL].max()
        min_time = cmin if min_time is None else min(min_time, cmin)
        max_time = cmax if max_time is None else max(max_time, cmax)

        part = (stable_hash_pair(chunk[CUST_COL], chunk[ACCT_COL]) % N_PARTITIONS).astype("int32")
        chunk["_part"] = part

        for p, sub in chunk.groupby("_part", sort=False):
            out_path = os.path.join(part_dir, f"part_{int(p):04d}_{chunk_id:06d}.parquet")
            sub = sub.drop(columns=["_part"])
            sub.to_parquet(out_path, index=False)

        chunk_id += 1

    if n_events_total == 0 or min_time is None or max_time is None:
        raise ValueError("No events processed. Check column names and delimiter.")

    global_end = max_time

    action_counts = None
    all_journeys = []
    all_gaps = []

    for p in range(N_PARTITIONS):
        paths = glob.glob(os.path.join(part_dir, f"part_{p:04d}_*.parquet"))
        if not paths:
            continue

        frames = [pd.read_parquet(x, columns=usecols) for x in paths]
        dfp = pd.concat(frames, ignore_index=True)
        if dfp.empty:
            continue

        vc = dfp.groupby([ACTION_ID_COL, ACTION_NAME_COL], dropna=False).size()
        action_counts = vc if action_counts is None else action_counts.add(vc, fill_value=0)

        dfp = prepare_partition_df(dfp)
        if dfp.empty:
            continue

        journeys_p, gaps_p_days = build_journeys_and_gaps(dfp, global_end=global_end)
        all_journeys.append(journeys_p)
        all_gaps.append(gaps_p_days)

    if not all_journeys:
        raise ValueError("No journeys produced. Check event names and timestamps.")

    journeys = pd.concat(all_journeys, ignore_index=True)
    gaps_days = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.Series(dtype="float64")

    outcome_counts = journeys["outcome"].value_counts().rename_axis("outcome").reset_index(name="count")
    outcome_counts["share"] = outcome_counts["count"] / outcome_counts["count"].sum()

    actions_count_stats = summarize_numeric(journeys["actions_count"], "actions_count")
    duration_stats = summarize_numeric(journeys["duration_days"], "duration_days")
    gap_stats = summarize_numeric(gaps_days, "gap_days")

    if action_counts is None:
        actions_tbl = pd.DataFrame(columns=[ACTION_ID_COL, ACTION_NAME_COL, "count", "share"])
    else:
        actions_tbl = (
            action_counts.reset_index(name="count")
                         .sort_values("count", ascending=False)
                         .head(25)
        )
        actions_tbl["share"] = actions_tbl["count"] / float(n_events_total)

    tmpdir = tempfile.mkdtemp(prefix="journey_eda_full_fast_")
    pngs = []

    p1 = os.path.join(tmpdir, "journey_actions_hist.png")
    save_hist_png(journeys["actions_count"], "Journey length (# actions)", "# actions", p1, bins=80, logy=True)
    pngs.append(p1)

    p2 = os.path.join(tmpdir, "journey_duration_hist.png")
    save_hist_png(journeys["duration_days"], "Journey duration (days)", "days", p2, bins=80, logy=True)
    pngs.append(p2)

    p3 = os.path.join(tmpdir, "gap_hist.png")
    save_hist_png(gaps_days, "Inter-action gaps (days)", "days", p3, bins=120, logy=True)
    pngs.append(p3)

    p4 = os.path.join(tmpdir, "outcome_bar.png")
    save_bar_png(outcome_counts["outcome"], outcome_counts["count"], "Journey outcomes", "Outcome", "Count", p4)
    pngs.append(p4)

    unc = journeys[journeys["outcome"].isin(["SUCCESS", "INACTIVITY"])].copy()

    p5 = os.path.join(tmpdir, "actions_success_vs_inactivity.png")
    plt.figure()
    plt.hist(unc.loc[unc["outcome"] == "SUCCESS", "actions_count"], bins=80, alpha=0.6, label="SUCCESS")
    plt.hist(unc.loc[unc["outcome"] == "INACTIVITY", "actions_count"], bins=80, alpha=0.6, label="INACTIVITY")
    plt.yscale("log")
    plt.title("Actions per journey: SUCCESS vs INACTIVITY")
    plt.xlabel("# actions")
    plt.ylabel("Count (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p5, dpi=200)
    plt.close()
    pngs.append(p5)

    p6 = os.path.join(tmpdir, "duration_success_vs_inactivity.png")
    plt.figure()
    plt.hist(unc.loc[unc["outcome"] == "SUCCESS", "duration_days"], bins=80, alpha=0.6, label="SUCCESS")
    plt.hist(unc.loc[unc["outcome"] == "INACTIVITY", "duration_days"], bins=80, alpha=0.6, label="INACTIVITY")
    plt.yscale("log")
    plt.title("Journey duration (days): SUCCESS vs INACTIVITY")
    plt.xlabel("days")
    plt.ylabel("Count (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p6, dpi=200)
    plt.close()
    pngs.append(p6)

    meta = {
        "data_path": DATA_PATH,
        "sep": sep,
        "n_events": int(n_events_total),
        "n_journeys": int(len(journeys)),
        "min_time": str(min_time),
        "max_time": str(max_time),
        "n_partitions": int(N_PARTITIONS),
    }

    build_pdf_report(
        output_pdf=OUTPUT_PDF,
        meta=meta,
        actions_tbl=actions_tbl,
        actions_count_stats=actions_count_stats,
        duration_stats=duration_stats,
        gap_stats=gap_stats,
        outcome_counts=outcome_counts,
        png_paths=pngs,
    )

    print("Saved PDF:", OUTPUT_PDF)


if __name__ == "__main__":
    main()
