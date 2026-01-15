from __future__ import annotations

import os
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


DATA_PATH = r"C:\Users\samvi\Downloads\dat_train1.csv"
PART_DIR = r"C:\Users\samvi\Downloads\m148_eda_work_full_fast\partitions"
OUTPUT_PDF = r"C:\Users\samvi\Downloads\journey_eda_duplicates_report.pdf"

TIME_COL = "event_timestamp"
CUST_COL = "customer_id"
ACCT_COL = "account_id"
ACTION_ID_COL = "ed_id"
ACTION_NAME_COL = "event_name"

SUCCESS_EVENT = "order_shipped"
INACTIVITY_DAYS = 60
INACTIVITY = pd.Timedelta(days=INACTIVITY_DAYS)


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def df_to_rl_table(df: pd.DataFrame, max_rows: int = 50, floatfmt: str = "{:.4g}") -> Table:
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
        "shipped_count": g[ACTION_NAME_COL].apply(lambda x: (x == SUCCESS_EVENT).sum()),
    })
    journeys["has_success"] = journeys["shipped_count"] > 0
    journeys["duration_days"] = (journeys["end_time"] - journeys["start_time"]).dt.total_seconds() / (24.0 * 3600.0)
    journeys["is_censored"] = (~journeys["has_success"]) & ((global_end - journeys["end_time"]) < INACTIVITY)
    journeys["outcome"] = np.where(journeys["has_success"], "SUCCESS",
                            np.where(journeys["is_censored"], "CENSORED", "INACTIVITY"))
    journeys = journeys.reset_index()

    gaps_days = (df.groupby("journey_id")[TIME_COL].diff().dt.total_seconds() / (24.0 * 3600.0)).dropna()
    return journeys, gaps_days


def build_pdf_report(
    output_pdf: str,
    meta: dict,
    dup_summary: pd.DataFrame,
    journeys_per_id_stats: pd.DataFrame,
    journeys_per_customer_stats: pd.DataFrame,
    outcome_counts: pd.DataFrame,
    duration_stats: pd.DataFrame,
    gap_stats: pd.DataFrame,
    png_paths: list[str],
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    story = []
    story.append(Paragraph("Unique IDs vs Journeys Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    meta_lines = [
        f"<b>Data file:</b> {meta['data_path']}",
        f"<b>Partitions folder:</b> {meta['part_dir']}",
        f"<b>Partition files:</b> {meta['n_part_files']:,}",
        f"<b>Events processed:</b> {meta['n_events']:,}",
        f"<b>Journeys:</b> {meta['n_journeys']:,}",
        f"<b>Inactivity threshold:</b> {INACTIVITY_DAYS} days",
        f"<b>Success event:</b> {SUCCESS_EVENT}",
        f"<b>Dataset window:</b> {meta['min_time']} to {meta['max_time']}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Summary: IDs and multi-journey duplication", styles["Heading2"]))
    story.append(df_to_rl_table(dup_summary, max_rows=50))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Journeys per (customer_id, account_id)", styles["Heading2"]))
    story.append(df_to_rl_table(journeys_per_id_stats, max_rows=50))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Journeys per customer_id", styles["Heading2"]))
    story.append(df_to_rl_table(journeys_per_customer_stats, max_rows=50))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Outcomes", styles["Heading2"]))
    story.append(df_to_rl_table(outcome_counts, max_rows=50))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Timing", styles["Heading2"]))
    story.append(Paragraph("<b>Journey duration (days)</b>", styles["Heading3"]))
    story.append(df_to_rl_table(duration_stats, max_rows=50))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("<b>Inter-action gaps (days)</b>", styles["Heading3"]))
    story.append(df_to_rl_table(gap_stats, max_rows=50))
    story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Figures", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    for p in png_paths:
        story.append(Image(p, width=7.2 * inch, height=4.2 * inch))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


def main() -> None:
    ensure_dir(OUTPUT_PDF)

    part_files = sorted(glob.glob(os.path.join(PART_DIR, "part_*.parquet")))
    if not part_files:
        raise ValueError(f"No parquet partitions found in: {PART_DIR}")

    usecols = [CUST_COL, ACCT_COL, ACTION_ID_COL, ACTION_NAME_COL, TIME_COL]

    n_events_total = 0
    min_time = None
    max_time = None

    for f in part_files:
        dfp = pd.read_parquet(f, columns=[TIME_COL])
        ts = pd.to_datetime(dfp[TIME_COL], errors="coerce", utc=True).dropna()
        if ts.empty:
            continue
        n_events_total += int(len(ts))
        cmin = ts.min()
        cmax = ts.max()
        min_time = cmin if min_time is None else min(min_time, cmin)
        max_time = cmax if max_time is None else max(max_time, cmax)

    if n_events_total == 0 or min_time is None or max_time is None:
        raise ValueError("Could not infer dataset window from partitions.")

    global_end = max_time

    all_journeys = []
    all_gaps = []

    for f in part_files:
        dfp = pd.read_parquet(f, columns=usecols)
        if dfp.empty:
            continue
        dfp = prepare_partition_df(dfp)
        if dfp.empty:
            continue
        journeys_p, gaps_p_days = build_journeys_and_gaps(dfp, global_end=global_end)
        all_journeys.append(journeys_p)
        all_gaps.append(gaps_p_days)

    if not all_journeys:
        raise ValueError("No journeys produced from partitions.")

    journeys = pd.concat(all_journeys, ignore_index=True)
    gaps_days = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.Series(dtype="float64")

    journeys_per_id = journeys.groupby([CUST_COL, ACCT_COL]).size()
    journeys_per_customer = journeys.groupby(CUST_COL).size()

    dup_summary = pd.DataFrame([
        {"metric": "journeys_total", "value": int(len(journeys))},
        {"metric": "unique_ids_customer_account", "value": int(journeys_per_id.shape[0])},
        {"metric": "avg_journeys_per_id", "value": float(journeys_per_id.mean())},
        {"metric": "median_journeys_per_id", "value": float(journeys_per_id.median())},
        {"metric": "ids_with_multiple_journeys", "value": int((journeys_per_id > 1).sum())},
        {"metric": "extra_journeys_beyond_1_per_id", "value": int(journeys_per_id.sum() - journeys_per_id.shape[0])},
        {"metric": "unique_customers", "value": int(journeys_per_customer.shape[0])},
        {"metric": "customers_with_multiple_journeys", "value": int((journeys_per_customer > 1).sum())},
        {"metric": "journeys_with_multiple_order_shipped", "value": int((journeys["shipped_count"] > 1).sum())},
    ])

    journeys_per_id_stats = summarize_numeric(journeys_per_id.astype(float), "journeys_per_id")
    journeys_per_customer_stats = summarize_numeric(journeys_per_customer.astype(float), "journeys_per_customer")

    outcome_counts = journeys["outcome"].value_counts().rename_axis("outcome").reset_index(name="count")
    outcome_counts["share"] = outcome_counts["count"] / outcome_counts["count"].sum()

    duration_stats = summarize_numeric(journeys["duration_days"], "duration_days")
    gap_stats = summarize_numeric(gaps_days, "gap_days")

    tmpdir = tempfile.mkdtemp(prefix="journey_dups_")
    pngs = []

    p1 = os.path.join(tmpdir, "journeys_per_id_hist.png")
    save_hist_png(journeys_per_id.astype(float), "Journeys per (customer_id, account_id)", "journeys per id", p1, bins=60, logy=True)
    pngs.append(p1)

    p2 = os.path.join(tmpdir, "journeys_per_customer_hist.png")
    save_hist_png(journeys_per_customer.astype(float), "Journeys per customer_id", "journeys per customer", p2, bins=60, logy=True)
    pngs.append(p2)

    p3 = os.path.join(tmpdir, "outcomes_bar.png")
    save_bar_png(outcome_counts["outcome"], outcome_counts["count"], "Journey outcomes", "Outcome", "Count", p3)
    pngs.append(p3)

    p4 = os.path.join(tmpdir, "duration_days_hist.png")
    save_hist_png(journeys["duration_days"], "Journey duration (days)", "days", p4, bins=80, logy=True)
    pngs.append(p4)

    p5 = os.path.join(tmpdir, "gap_days_hist.png")
    save_hist_png(gaps_days, "Inter-action gaps (days)", "days", p5, bins=120, logy=True)
    pngs.append(p5)

    meta = {
        "data_path": DATA_PATH,
        "part_dir": PART_DIR,
        "n_part_files": int(len(part_files)),
        "n_events": int(n_events_total),
        "n_journeys": int(len(journeys)),
        "min_time": str(min_time),
        "max_time": str(max_time),
    }

    build_pdf_report(
        output_pdf=OUTPUT_PDF,
        meta=meta,
        dup_summary=dup_summary,
        journeys_per_id_stats=journeys_per_id_stats,
        journeys_per_customer_stats=journeys_per_customer_stats,
        outcome_counts=outcome_counts,
        duration_stats=duration_stats,
        gap_stats=gap_stats,
        png_paths=pngs,
    )

    print("Saved PDF:", OUTPUT_PDF)


if __name__ == "__main__":
    main()
