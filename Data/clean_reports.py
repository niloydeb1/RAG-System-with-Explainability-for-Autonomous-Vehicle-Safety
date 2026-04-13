from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =========================
# CONFIG
# =========================
INPUT_FILES = [
    "raw_pre_mid_2025.csv",
    "raw_post_mid_2025.csv",
]

OUTPUT_DIR = Path("clean_reports_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Safe missing-value tokens only.
# Do NOT include things like "Unknown, see Narrative" because those are meaningful values.
MISSING_TOKENS = {
    "",
    " ",
    "  ",
    "   ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "None",
}

# Columns that should usually be numeric if present
NUMERIC_COLUMNS = [
    "Report Version",
    "Report Month",
    "Report Year",
    "Model Year",
    "Mileage",
    "Posted Speed Limit (MPH)",
    "SV Precrash Speed (MPH)",
]

# Columns that should be parsed as dates if present
DATE_COLUMNS = [
    "Report Submission Date",
    "Notice Received Date",
    "Incident Date",
]

# Columns that should be normalized as uppercase state abbreviations if present
STATE_COLUMNS = ["State"]

# These are useful to keep as text but should be whitespace-trimmed
TEXT_COLUMNS_ALWAYS_STRIP = [
    "Report ID",
    "Reporting Entity",
    "Report Type",
    "VIN",
    "Serial Number",
    "Make",
    "Model",
    "Same Vehicle ID",
    "Driver / Operator Type",
    "Automation Feature Version",
    "ADAS/ADS System Version",
    "ADAS/ADS Hardware Version",
    "ADAS/ADS Software Version",
    "Automation System Engaged?",
    "Engagement Status",
    "Operating Entity",
    "Same Incident ID",
    "Incident Time (24:00)",
    "Address",
    "City",
    "Zip Code",
    "Roadway Type",
    "Roadway Surface",
    "Roadway Description",
    "Lighting",
    "Crash With",
    "Highest Injury Severity Alleged",
    "CP Pre-Crash Movement",
    "SV Pre-Crash Movement",
    "Investigating Agency",
    "Narrative",
    "Within ODD?",
]


# =========================
# HELPERS
# =========================
def detect_schema_version(columns: List[str]) -> str:
    """
    Detect schema by distinctive columns.

    post_mid_2025 (newer schema) usually has:
      - Automation Feature Version
      - VIN Decoded
      - Engagement Status
      - roadway checkbox set like 'Roadway-Work Zone'

    pre_mid_2025 (older schema) usually has:
      - ADAS/ADS System Version
      - ADS Equipped?
      - Mileage
      - Weather - Other Text
    """
    colset = set(columns)

    if "Automation Feature Version" in colset or "VIN Decoded" in colset:
        return "post_mid_2025"

    if "ADAS/ADS System Version" in colset or "ADS Equipped?" in colset:
        return "pre_mid_2025"

    return "unknown"


def strip_and_standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        # Convert to string only where non-null, trim whitespace
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert safe missing tokens to NA
        df[col] = df[col].replace(list(MISSING_TOKENS), pd.NA)

    return df


def normalize_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in STATE_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .map(lambda x: x.strip().upper() if isinstance(x, str) else x)
            )

    return df


def normalize_time_string(value):
    """
    Keep times as strings because NHTSA includes values like 24:00.
    We do light normalization only.
    """
    if pd.isna(value):
        return pd.NA

    if not isinstance(value, str):
        value = str(value).strip()

    value = value.strip()

    if value in MISSING_TOKENS:
        return pd.NA

    # Handle simple formats like 7:5 -> 07:05
    if ":" in value:
        parts = value.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            hh = parts[0].zfill(2)
            mm = parts[1].zfill(2)
            return f"{hh}:{mm}"

    return value


def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Incident Time (24:00)" in df.columns:
        df["Incident Time (24:00)"] = df["Incident Time (24:00)"].map(normalize_time_string)

    return df


def parse_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def parse_month_year_value(value):
    """
    Parse NHTSA month-year values like JAN-2026 or JUN-2025
    and return YYYY-MM.

    Also safely handles already-clean values like YYYY-MM or YYYY-MM-DD.
    """
    if pd.isna(value):
        return pd.NA

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value in MISSING_TOKENS:
        return pd.NA

    # Already in YYYY-MM
    if len(value) == 7 and value[4] == "-":
        return value

    # Already in YYYY-MM-DD -> reduce to YYYY-MM
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:7]

    # Main expected raw format: JAN-2026
    try:
        dt = pd.to_datetime(value.title(), format="%b-%Y", errors="raise")
        return dt.strftime("%Y-%m")
    except Exception:
        pass

    # Fallback for possible 2-digit year formats like JAN-26
    try:
        dt = pd.to_datetime(value.title(), format="%b-%y", errors="raise")
        return dt.strftime("%Y-%m")
    except Exception:
        pass

    # Final fallback: let pandas try, then reduce to YYYY-MM
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return pd.NA
        return dt.strftime("%Y-%m")
    except Exception:
        return pd.NA

def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(parse_month_year_value)

    return df

def remove_non_incident_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df = df.copy()

    if "Report Type" not in df.columns:
        return df, 0

    before = len(df)
    mask = df["Report Type"].eq("No New or Updated Incident Reports")
    removed = int(mask.fillna(False).sum())
    df = df.loc[~mask.fillna(False)].copy()

    return df, removed


def keep_latest_report_version(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Keep only the latest Report Version per Report ID.
    If there is still a tie, keep the row that appears last after sorting by:
      - Report ID
      - Report Version
      - Report Submission Date
    """
    df = df.copy()

    if "Report ID" not in df.columns or "Report Version" not in df.columns:
        return df, 0

    before = len(df)

    # Temporary sortable date
    if "Report Submission Date" in df.columns:
        sort_date = pd.to_datetime(df["Report Submission Date"], errors="coerce")
    else:
        sort_date = pd.Series(pd.NaT, index=df.index)

    df["_sort_date_tmp"] = sort_date
    df["_row_order_tmp"] = range(len(df))

    df = df.sort_values(
        by=["Report ID", "Report Version", "_sort_date_tmp", "_row_order_tmp"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    df = df.drop_duplicates(subset=["Report ID"], keep="last").copy()

    removed = before - len(df)

    df = df.drop(columns=["_sort_date_tmp", "_row_order_tmp"], errors="ignore")

    return df, removed


def add_provenance_columns(df: pd.DataFrame, source_file: str, schema_version: str) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = source_file
    df["schema_version"] = schema_version
    return df


def build_summary(
        source_file: str,
        schema_version: str,
        rows_before: int,
        rows_after_non_incident_filter: int,
        rows_after_latest_version: int,
        removed_non_incident_rows: int,
        removed_older_versions: int,
        df_final: pd.DataFrame,
) -> Dict:
    summary = {
        "source_file": source_file,
        "schema_version": schema_version,
        "rows_before": rows_before,
        "rows_after_non_incident_filter": rows_after_non_incident_filter,
        "rows_after_latest_version": rows_after_latest_version,
        "removed_non_incident_rows": removed_non_incident_rows,
        "removed_older_versions": removed_older_versions,
        "n_columns_final": int(df_final.shape[1]),
        "n_unique_report_id": int(df_final["Report ID"].nunique()) if "Report ID" in df_final.columns else None,
        "n_unique_same_incident_id": int(df_final["Same Incident ID"].nunique(dropna=True)) if "Same Incident ID" in df_final.columns else None,
        "report_type_counts_final": (
            df_final["Report Type"].value_counts(dropna=False).to_dict()
            if "Report Type" in df_final.columns
            else {}
        ),
        "top_reporting_entities_final": (
            df_final["Reporting Entity"].value_counts(dropna=False).head(15).to_dict()
            if "Reporting Entity" in df_final.columns
            else {}
        ),
        "incident_date_min": (
            str(df_final["Incident Date"].dropna().min())
            if "Incident Date" in df_final.columns and df_final["Incident Date"].notna().any()
            else None
        ),
        "incident_date_max": (
            str(df_final["Incident Date"].dropna().max())
            if "Incident Date" in df_final.columns and df_final["Incident Date"].notna().any()
            else None
        ),
    }
    return summary


def clean_one_file(input_path: Path) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(input_path, dtype=object)

    rows_before = len(df)
    schema_version = detect_schema_version(df.columns.tolist())

    df = strip_and_standardize_strings(df)
    df = normalize_state_columns(df)
    df = normalize_time_columns(df)
    df = parse_numeric_columns(df)
    df = parse_date_columns(df)

    df, removed_non_incident_rows = remove_non_incident_rows(df)
    rows_after_non_incident_filter = len(df)

    df, removed_older_versions = keep_latest_report_version(df)
    rows_after_latest_version = len(df)

    df = add_provenance_columns(df, source_file=input_path.name, schema_version=schema_version)

    # Optional: move provenance columns to front
    front_cols = ["schema_version", "source_file"]
    rest_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + rest_cols]

    summary = build_summary(
        source_file=input_path.name,
        schema_version=schema_version,
        rows_before=rows_before,
        rows_after_non_incident_filter=rows_after_non_incident_filter,
        rows_after_latest_version=rows_after_latest_version,
        removed_non_incident_rows=removed_non_incident_rows,
        removed_older_versions=removed_older_versions,
        df_final=df,
    )

    return df, summary


# =========================
# MAIN
# =========================
def main():
    all_summaries = []

    for file_name in INPUT_FILES:
        input_path = Path(file_name)

        if not input_path.exists():
            print(f"[WARNING] File not found, skipping: {input_path}")
            continue

        df_clean, summary = clean_one_file(input_path)

        schema_version = summary["schema_version"]

        if schema_version == "pre_mid_2025":
            out_csv = OUTPUT_DIR / "clean_pre_mid_2025_reports.csv"
        elif schema_version == "post_mid_2025":
            out_csv = OUTPUT_DIR / "clean_post_mid_2025_reports.csv"
        else:
            out_csv = OUTPUT_DIR / f"clean_unknown_schema_{input_path.stem}.csv"

        df_clean.to_csv(out_csv, index=False)
        print(f"[OK] Wrote: {out_csv}")

        summary_path = OUTPUT_DIR / f"{out_csv.stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[OK] Wrote: {summary_path}")
        all_summaries.append(summary)

    combined_summary_path = OUTPUT_DIR / "cleaning_summary_all.json"
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"[OK] Wrote: {combined_summary_path}")


if __name__ == "__main__":
    main()