from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


INPUT_FILE = "harmonized_reports_latest.csv"
OUTPUT_DIR = Path("incident_cases_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------
# basic helpers
# ---------------------------
UNKNOWN_LIKE = {
    "unknown",
    "unk",
    "unknown, see narrative",
    "unknown - see narrative",
}


def clean_text(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def normalize_text_for_choice(x: Any) -> Optional[str]:
    s = clean_text(x)
    if s is None:
        return None
    return s


def parse_json_list(x: Any) -> List[str]:
    s = clean_text(x)
    if s is None:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass

    return [s]


def safe_int(x: Any) -> Optional[int]:
    s = clean_text(x)
    if s is None:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def info_score(value: Any) -> float:
    """
    Higher score = more informative / more preferable.
    """
    s = clean_text(value)
    if s is None:
        return -1.0

    low = s.lower()
    score = 1.0

    if low in UNKNOWN_LIKE:
        score -= 2.0

    if "see narrative" in low:
        score -= 0.5

    score += min(len(s) / 80.0, 3.0)
    return score


def choose_best_scalar(series: pd.Series) -> Optional[str]:
    """
    Pick one best scalar value from a group:
    - ignore blanks
    - prefer informative values over Unknown-like values
    - prefer repeated values if multiple reports agree
    """
    vals = [normalize_text_for_choice(v) for v in series]
    vals = [v for v in vals if v is not None]

    if not vals:
        return None

    informative = [v for v in vals if v.lower() not in UNKNOWN_LIKE]
    pool = informative if informative else vals

    counts = Counter(pool)

    best = max(
        counts.keys(),
        key=lambda v: (counts[v], info_score(v), len(v)),
    )
    return best


def union_json_list_column(series: pd.Series) -> Optional[str]:
    merged: List[str] = []
    for v in series:
        merged.extend(parse_json_list(v))

    merged = sorted(set(merged))
    return json.dumps(merged, ensure_ascii=False) if merged else None


def min_nonnull_ym(series: pd.Series) -> Optional[str]:
    vals = [clean_text(v) for v in series]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def max_nonnull_ym(series: pd.Series) -> Optional[str]:
    vals = [clean_text(v) for v in series]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def min_nonnull_int(series: pd.Series) -> Optional[int]:
    vals = [safe_int(v) for v in series]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def max_nonnull_int(series: pd.Series) -> Optional[int]:
    vals = [safe_int(v) for v in series]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# ---------------------------
# narrative selection
# ---------------------------
BOILERPLATE_PHRASES = [
    "duplicate of the redacted public copy",
    "submitting a duplicate",
    "this report is a transdev duplicate",
    "reference file:",
    "has not investigated the alleged incident",
    "incorporates cruise's responses into this submission by reference",
    "submission is based on facts supplied",
]


def narrative_score(text: Any) -> float:
    s = clean_text(text)
    if s is None:
        return -1e9

    low = s.lower()
    score = float(len(s))

    for phrase in BOILERPLATE_PHRASES:
        if phrase in low:
            score -= 500.0

    if "[redacted" in low:
        score -= 100.0

    return score


def choose_best_narrative(group: pd.DataFrame) -> Dict[str, Optional[str]]:
    best_idx = None
    best_score = -1e18

    for idx, row in group.iterrows():
        score = narrative_score(row.get("narrative"))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return {
            "best_narrative": None,
            "best_narrative_report_id": None,
            "best_narrative_reporting_entity": None,
        }

    row = group.loc[best_idx]
    return {
        "best_narrative": clean_text(row.get("narrative")),
        "best_narrative_report_id": clean_text(row.get("report_id")),
        "best_narrative_reporting_entity": clean_text(row.get("reporting_entity")),
    }


# ---------------------------
# main aggregation logic
# ---------------------------
SCALAR_COLUMNS_TO_SELECT = [
    "vehicle_make",
    "vehicle_model",
    "vehicle_model_year",
    "mileage",
    "driver_operator_type",
    "operating_entity",
    "automation_type_engaged",
    "engagement_status",
    "within_odd",
    "automation_feature_version",
    "automation_system_version",
    "automation_hardware_version",
    "automation_software_version",
    "ads_equipped",
    "incident_ym",
    "incident_time_local",
    "city",
    "state",
    "latitude",
    "longitude",
    "roadway_type",
    "roadway_surface_text",
    "roadway_description_text",
    "lighting",
    "posted_speed_limit_mph",
    "weather_other_text",
    "crash_with",
    "injury_severity",
    "property_damage",
    "cp_pre_crash_movement",
    "sv_pre_crash_movement",
    "sv_precrash_speed_mph",
    "any_airbags_deployed",
    "any_vehicle_towed",
    "all_passengers_belted",
    "investigating_agency",
    "law_enforcement_investigating",
    "rep_ent_or_mfr_investigating",
]

JSON_LIST_COLUMNS_TO_UNION = [
    "roadway_conditions_json",
    "weather_conditions_json",
    "cp_contact_areas_json",
    "sv_contact_areas_json",
    "notice_sources_json",
    "available_evidence_json",
]


def build_incident_key(row: pd.Series) -> str:
    same_incident_id = clean_text(row.get("same_incident_id"))
    report_id = clean_text(row.get("report_id"))

    if same_incident_id is not None:
        return same_incident_id

    return f"NO_SAME_INCIDENT__{report_id}"


def aggregate_one_incident(group: pd.DataFrame) -> Dict[str, Any]:
    incident_case_id = clean_text(group["incident_case_id"].iloc[0])

    out: Dict[str, Any] = {
        "incident_case_id": incident_case_id,
        "same_incident_id": None if incident_case_id.startswith("NO_SAME_INCIDENT__") else incident_case_id,
        "report_count": int(len(group)),
        "report_ids_json": json.dumps(
            sorted([clean_text(v) for v in group["report_id"] if clean_text(v) is not None]),
            ensure_ascii=False,
        ),
        "reporting_entities_json": json.dumps(
            sorted(set([clean_text(v) for v in group["reporting_entity"] if clean_text(v) is not None])),
            ensure_ascii=False,
        ),
        "schema_versions_json": json.dumps(
            sorted(set([clean_text(v) for v in group["schema_version"] if clean_text(v) is not None])),
            ensure_ascii=False,
        ),
        "source_files_json": json.dumps(
            sorted(set([clean_text(v) for v in group["source_file"] if clean_text(v) is not None])),
            ensure_ascii=False,
        ),
        "report_submission_ym_min": min_nonnull_ym(group["report_submission_ym"]),
        "report_submission_ym_max": max_nonnull_ym(group["report_submission_ym"]),
        "report_month_min": min_nonnull_int(group["report_month"]),
        "report_month_max": max_nonnull_int(group["report_month"]),
        "report_year_min": min_nonnull_int(group["report_year"]),
        "report_year_max": max_nonnull_int(group["report_year"]),
        "all_narratives_count": int(sum(clean_text(v) is not None for v in group["narrative"])),
        "narrative_redacted_any": bool(
            any(str(v).strip().lower() == "true" for v in group["narrative_redacted"].fillna("").astype(str))
        ),
    }

    # choose best scalar values
    for col in SCALAR_COLUMNS_TO_SELECT:
        out[col] = choose_best_scalar(group[col]) if col in group.columns else None

    # union list-style JSON columns
    for col in JSON_LIST_COLUMNS_TO_UNION:
        out[col] = union_json_list_column(group[col]) if col in group.columns else None

    # choose best narrative
    out.update(choose_best_narrative(group))

    # keep lightweight report-level provenance
    out["report_links_json"] = json.dumps(
        [
            {
                "report_id": clean_text(row.get("report_id")),
                "reporting_entity": clean_text(row.get("reporting_entity")),
                "report_submission_ym": clean_text(row.get("report_submission_ym")),
                "schema_version": clean_text(row.get("schema_version")),
            }
            for _, row in group.iterrows()
        ],
        ensure_ascii=False,
    )

    return out


def main():
    df = pd.read_csv(INPUT_FILE, dtype=object)

    # build grouping key
    df["incident_case_id"] = df.apply(build_incident_key, axis=1)

    records: List[Dict[str, Any]] = []
    for _, group in df.groupby("incident_case_id", sort=False):
        records.append(aggregate_one_incident(group))

    incident_cases = pd.DataFrame(records)

    # column order
    front_cols = [
        "incident_case_id",
        "same_incident_id",
        "report_count",
        "report_ids_json",
        "reporting_entities_json",
        "schema_versions_json",
        "source_files_json",
        "report_submission_ym_min",
        "report_submission_ym_max",
        "report_month_min",
        "report_month_max",
        "report_year_min",
        "report_year_max",
    ]

    middle_cols = SCALAR_COLUMNS_TO_SELECT + JSON_LIST_COLUMNS_TO_UNION

    tail_cols = [
        "best_narrative",
        "best_narrative_report_id",
        "best_narrative_reporting_entity",
        "all_narratives_count",
        "narrative_redacted_any",
        "report_links_json",
    ]

    ordered_cols = front_cols + middle_cols + tail_cols
    incident_cases = incident_cases[ordered_cols]

    out_csv = OUTPUT_DIR / "incident_cases.csv"
    incident_cases.to_csv(out_csv, index=False)

    summary = {
        "input_rows_report_level": int(len(df)),
        "output_rows_incident_level": int(len(incident_cases)),
        "unique_same_incident_id_input": int(df["same_incident_id"].nunique(dropna=True)),
        "multi_report_incident_cases": int((incident_cases["report_count"] > 1).sum()),
        "max_reports_in_one_incident_case": int(incident_cases["report_count"].max()),
        "fallback_cases_without_same_incident_id": int(incident_cases["same_incident_id"].isna().sum()),
    }

    with open(OUTPUT_DIR / "incident_cases_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] wrote", out_csv)
    print("[OK] wrote", OUTPUT_DIR / "incident_cases_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()