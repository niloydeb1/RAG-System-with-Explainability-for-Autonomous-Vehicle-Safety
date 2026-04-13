from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


INPUT_FILE = "incident_cases.csv"   # or incident_cases_stage5.csv if that is your filename
OUTPUT_DIR = Path("rag_documents_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# helpers
# -----------------------------
def clean_text(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def clean_bool(x: Any) -> Optional[bool]:
    s = clean_text(x)
    if s is None:
        return None
    low = s.lower()
    if low in {"true", "1", "yes", "y"}:
        return True
    if low in {"false", "0", "no", "n"}:
        return False
    return None


def clean_int(x: Any) -> Optional[int]:
    s = clean_text(x)
    if s is None:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def clean_float(x: Any) -> Optional[float]:
    s = clean_text(x)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_json_list(x: Any) -> List[str]:
    s = clean_text(x)
    if s is None:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for v in obj:
                vv = clean_text(v)
                if vv is not None:
                    out.append(vv)
            return out
    except Exception:
        pass
    return [s]


def list_to_pipe_string(values: List[str]) -> Optional[str]:
    values = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    values = sorted(set(values))
    return "|".join(values) if values else None


def format_list_for_text(values: List[str]) -> Optional[str]:
    values = [v.replace("_", " ") for v in values if isinstance(v, str) and v.strip()]
    values = sorted(set(values))
    return ", ".join(values) if values else None


def ym_to_year(ym: Optional[str]) -> Optional[int]:
    if ym is None:
        return None
    if len(ym) >= 4 and ym[:4].isdigit():
        return int(ym[:4])
    return None


def normalize_unknownish(value: Optional[str]) -> Optional[str]:
    s = clean_text(value)
    if s is None:
        return None

    low = s.lower()
    if low in {"unknown", "unk"}:
        return "Unknown"
    if "unknown" in low:
        return s
    return s


def first_from_json_list_field(x: Any) -> Optional[str]:
    vals = parse_json_list(x)
    return vals[0] if vals else None


# -----------------------------
# feature extraction
# -----------------------------
def build_filter_metadata(row: pd.Series) -> Dict[str, Any]:
    weather_conditions = parse_json_list(row.get("weather_conditions_json"))
    roadway_conditions = parse_json_list(row.get("roadway_conditions_json"))
    evidence = parse_json_list(row.get("available_evidence_json"))
    notice_sources = parse_json_list(row.get("notice_sources_json"))
    reporting_entities = parse_json_list(row.get("reporting_entities_json"))
    schema_versions = parse_json_list(row.get("schema_versions_json"))
    cp_contact_areas = parse_json_list(row.get("cp_contact_areas_json"))
    sv_contact_areas = parse_json_list(row.get("sv_contact_areas_json"))

    incident_ym = clean_text(row.get("incident_ym"))
    report_submission_ym_max = clean_text(row.get("report_submission_ym_max"))

    metadata = {
        "document_id": f"incident#{clean_text(row.get('incident_case_id'))}",
        "chunk_number": 1,
        "incident_case_id": clean_text(row.get("incident_case_id")),
        "same_incident_id": clean_text(row.get("same_incident_id")),
        "report_count": clean_int(row.get("report_count")),
        "multi_report_case": (clean_int(row.get("report_count")) or 0) > 1,

        "incident_ym": incident_ym,
        "incident_year": ym_to_year(incident_ym),
        "report_submission_ym_max": report_submission_ym_max,
        "report_submission_year_max": ym_to_year(report_submission_ym_max),

        "state": clean_text(row.get("state")),
        "city": clean_text(row.get("city")),
        "roadway_type": clean_text(row.get("roadway_type")),
        "crash_with": clean_text(row.get("crash_with")),
        "injury_severity": clean_text(row.get("injury_severity")),
        "automation_type_engaged": clean_text(row.get("automation_type_engaged")),
        "engagement_status": clean_text(row.get("engagement_status")),
        "within_odd": clean_text(row.get("within_odd")),
        "driver_operator_type": clean_text(row.get("driver_operator_type")),
        "vehicle_make": clean_text(row.get("vehicle_make")),
        "vehicle_model": clean_text(row.get("vehicle_model")),
        "vehicle_model_year": clean_int(row.get("vehicle_model_year")),

        "primary_reporting_entity": clean_text(row.get("best_narrative_reporting_entity")) or first_from_json_list_field(row.get("reporting_entities_json")),
        "reporting_entities": list_to_pipe_string(reporting_entities),
        "schema_versions": list_to_pipe_string(schema_versions),

        "weather_conditions": list_to_pipe_string(weather_conditions),
        "roadway_conditions": list_to_pipe_string(roadway_conditions),
        "cp_contact_areas": list_to_pipe_string(cp_contact_areas),
        "sv_contact_areas": list_to_pipe_string(sv_contact_areas),

        "has_video_evidence": "video" in evidence,
        "has_police_report_evidence": "police_report" in evidence,
        "has_telematics_evidence": "telematics" in evidence,
        "has_edr_evidence": "edr" in evidence,
        "has_complaints_evidence": "complaints" in evidence,
        "notice_sources": list_to_pipe_string(notice_sources),

        "narrative_redacted_any": clean_bool(row.get("narrative_redacted_any")),
        "all_narratives_count": clean_int(row.get("all_narratives_count")),

        "best_narrative_report_id": clean_text(row.get("best_narrative_report_id")),
        "best_narrative_reporting_entity": clean_text(row.get("best_narrative_reporting_entity")),
    }

    return metadata


def build_document_title(row: pd.Series) -> str:
    incident_ym = clean_text(row.get("incident_ym")) or "unknown date"
    state = clean_text(row.get("state")) or "unknown state"
    crash_with = clean_text(row.get("crash_with")) or "unknown counterpart"
    automation = clean_text(row.get("automation_type_engaged")) or "automation unknown"
    return f"AV incident {incident_ym} | {state} | {crash_with} | {automation}"


def build_chunk_text(row: pd.Series) -> str:
    reporting_entities = parse_json_list(row.get("reporting_entities_json"))
    roadway_conditions = parse_json_list(row.get("roadway_conditions_json"))
    weather_conditions = parse_json_list(row.get("weather_conditions_json"))
    evidence = parse_json_list(row.get("available_evidence_json"))
    notice_sources = parse_json_list(row.get("notice_sources_json"))
    cp_contact_areas = parse_json_list(row.get("cp_contact_areas_json"))
    sv_contact_areas = parse_json_list(row.get("sv_contact_areas_json"))

    parts: List[str] = []

    incident_case_id = clean_text(row.get("incident_case_id"))
    same_incident_id = clean_text(row.get("same_incident_id"))
    report_count = clean_int(row.get("report_count"))

    parts.append(f"Incident case ID: {incident_case_id}.")
    if same_incident_id:
        parts.append(f"Same incident ID: {same_incident_id}.")
    if report_count is not None:
        parts.append(f"This case aggregates {report_count} report(s).")

    # time/location
    time_bits = []
    if clean_text(row.get("incident_ym")):
        time_bits.append(f"incident month: {clean_text(row.get('incident_ym'))}")
    if clean_text(row.get("incident_time_local")):
        time_bits.append(f"incident time: {clean_text(row.get('incident_time_local'))}")
    if time_bits:
        parts.append("Time context: " + "; ".join(time_bits) + ".")

    loc_bits = []
    if clean_text(row.get("city")):
        loc_bits.append(clean_text(row.get("city")))
    if clean_text(row.get("state")):
        loc_bits.append(clean_text(row.get("state")))
    if loc_bits:
        parts.append("Location: " + ", ".join(loc_bits) + ".")

    # reporting / vehicle
    if reporting_entities:
        parts.append("Reporting entities: " + ", ".join(sorted(set(reporting_entities))) + ".")

    vehicle_bits = []
    if clean_text(row.get("vehicle_make")):
        vehicle_bits.append(f"make: {clean_text(row.get('vehicle_make'))}")
    if clean_text(row.get("vehicle_model")):
        vehicle_bits.append(f"model: {clean_text(row.get('vehicle_model'))}")
    if clean_text(row.get("vehicle_model_year")):
        vehicle_bits.append(f"model year: {clean_text(row.get('vehicle_model_year'))}")
    if vehicle_bits:
        parts.append("Vehicle: " + "; ".join(vehicle_bits) + ".")

    # automation context
    auto_bits = []
    if clean_text(row.get("automation_type_engaged")):
        auto_bits.append(f"automation engaged: {clean_text(row.get('automation_type_engaged'))}")
    if clean_text(row.get("engagement_status")):
        auto_bits.append(f"engagement status: {clean_text(row.get('engagement_status'))}")
    if clean_text(row.get("within_odd")):
        auto_bits.append(f"within ODD: {clean_text(row.get('within_odd'))}")
    if clean_text(row.get("driver_operator_type")):
        auto_bits.append(f"driver/operator type: {clean_text(row.get('driver_operator_type'))}")
    if clean_text(row.get("operating_entity")):
        auto_bits.append(f"operating entity: {clean_text(row.get('operating_entity'))}")
    if auto_bits:
        parts.append("Automation context: " + "; ".join(auto_bits) + ".")

    # roadway / weather
    roadway_bits = []
    if clean_text(row.get("roadway_type")):
        roadway_bits.append(f"roadway type: {clean_text(row.get('roadway_type'))}")
    if clean_text(row.get("roadway_surface_text")):
        roadway_bits.append(f"roadway surface: {clean_text(row.get('roadway_surface_text'))}")
    if clean_text(row.get("roadway_description_text")):
        roadway_bits.append(f"roadway description: {clean_text(row.get('roadway_description_text'))}")
    if clean_text(row.get("lighting")):
        roadway_bits.append(f"lighting: {clean_text(row.get('lighting'))}")
    if clean_text(row.get("posted_speed_limit_mph")):
        roadway_bits.append(f"posted speed limit mph: {clean_text(row.get('posted_speed_limit_mph'))}")
    if roadway_conditions:
        roadway_bits.append(f"roadway conditions: {format_list_for_text(roadway_conditions)}")
    if roadway_bits:
        parts.append("Roadway context: " + "; ".join(roadway_bits) + ".")

    weather_bits = []
    if weather_conditions:
        weather_bits.append(f"weather conditions: {format_list_for_text(weather_conditions)}")
    if clean_text(row.get("weather_other_text")):
        weather_bits.append(f"weather other text: {clean_text(row.get('weather_other_text'))}")
    if weather_bits:
        parts.append("Weather context: " + "; ".join(weather_bits) + ".")

    # crash mechanics
    crash_bits = []
    if clean_text(row.get("crash_with")):
        crash_bits.append(f"crash with: {clean_text(row.get('crash_with'))}")
    if clean_text(row.get("sv_pre_crash_movement")):
        crash_bits.append(f"subject vehicle movement: {clean_text(row.get('sv_pre_crash_movement'))}")
    if clean_text(row.get("cp_pre_crash_movement")):
        crash_bits.append(f"crash partner movement: {clean_text(row.get('cp_pre_crash_movement'))}")
    if clean_text(row.get("sv_precrash_speed_mph")):
        crash_bits.append(f"subject vehicle pre-crash speed mph: {clean_text(row.get('sv_precrash_speed_mph'))}")
    if cp_contact_areas:
        crash_bits.append(f"crash partner contact areas: {format_list_for_text(cp_contact_areas)}")
    if sv_contact_areas:
        crash_bits.append(f"subject vehicle contact areas: {format_list_for_text(sv_contact_areas)}")
    if crash_bits:
        parts.append("Crash mechanics: " + "; ".join(crash_bits) + ".")

    # outcomes
    outcome_bits = []
    if clean_text(row.get("injury_severity")):
        outcome_bits.append(f"injury severity: {clean_text(row.get('injury_severity'))}")
    if clean_text(row.get("property_damage")):
        outcome_bits.append(f"property damage: {clean_text(row.get('property_damage'))}")
    if clean_text(row.get("any_airbags_deployed")):
        outcome_bits.append(f"any airbags deployed: {clean_text(row.get('any_airbags_deployed'))}")
    if clean_text(row.get("any_vehicle_towed")):
        outcome_bits.append(f"any vehicle towed: {clean_text(row.get('any_vehicle_towed'))}")
    if clean_text(row.get("all_passengers_belted")):
        outcome_bits.append(f"all passengers belted: {clean_text(row.get('all_passengers_belted'))}")
    if outcome_bits:
        parts.append("Outcomes: " + "; ".join(outcome_bits) + ".")

    # evidence / provenance
    prov_bits = []
    if evidence:
        prov_bits.append(f"available evidence: {format_list_for_text(evidence)}")
    if notice_sources:
        prov_bits.append(f"notice sources: {format_list_for_text(notice_sources)}")
    if clean_text(row.get("investigating_agency")):
        prov_bits.append(f"investigating agency: {clean_text(row.get('investigating_agency'))}")
    if clean_text(row.get("best_narrative_report_id")):
        prov_bits.append(f"best narrative source report ID: {clean_text(row.get('best_narrative_report_id'))}")
    if clean_text(row.get("best_narrative_reporting_entity")):
        prov_bits.append(f"best narrative source entity: {clean_text(row.get('best_narrative_reporting_entity'))}")
    if prov_bits:
        parts.append("Evidence and provenance: " + "; ".join(prov_bits) + ".")

    # narrative
    narrative = clean_text(row.get("best_narrative"))
    if narrative:
        parts.append("Narrative: " + narrative)

    return "\n".join(parts)


def build_record(row: pd.Series) -> Dict[str, Any]:
    incident_case_id = clean_text(row.get("incident_case_id"))
    record_id = f"incident#{incident_case_id}#chunk1"

    metadata = build_filter_metadata(row)
    title = build_document_title(row)
    chunk_text = build_chunk_text(row)

    # Human-inspectable row
    doc_row = {
        "_id": record_id,
        "document_id": metadata["document_id"],
        "chunk_number": 1,
        "document_title": title,
        "chunk_text": chunk_text,
        **metadata,
    }

    # Pinecone integrated-embedding record:
    # _id + chunk_text + flat metadata fields
    pinecone_record = {
        "_id": record_id,
        "chunk_text": chunk_text,
        "document_title": title,
        **metadata,
    }

    return {"doc_row": doc_row, "pinecone_record": pinecone_record}


def main():
    df = pd.read_csv(INPUT_FILE, dtype=object)

    doc_rows: List[Dict[str, Any]] = []
    pinecone_records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        built = build_record(row)
        doc_rows.append(built["doc_row"])
        pinecone_records.append(built["pinecone_record"])

    rag_df = pd.DataFrame(doc_rows)

    # Write inspectable CSV
    rag_csv = OUTPUT_DIR / "rag_documents.csv"
    rag_df.to_csv(rag_csv, index=False)

    # Write JSONL for Pinecone integrated embedding
    jsonl_path = OUTPUT_DIR / "pinecone_integrated_records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in pinecone_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Optional smaller preview CSV with only the most important fields
    preview_cols = [
        "_id",
        "document_title",
        "incident_case_id",
        "same_incident_id",
        "report_count",
        "incident_ym",
        "state",
        "city",
        "automation_type_engaged",
        "within_odd",
        "roadway_type",
        "crash_with",
        "injury_severity",
        "primary_reporting_entity",
        "has_video_evidence",
        "has_telematics_evidence",
        "chunk_text",
    ]
    preview_df = rag_df[preview_cols].copy()
    preview_path = OUTPUT_DIR / "rag_documents_preview.csv"
    preview_df.to_csv(preview_path, index=False)

    summary = {
        "input_incident_cases": int(len(df)),
        "output_rag_documents": int(len(rag_df)),
        "output_pinecone_records": int(len(pinecone_records)),
        "multi_report_cases_in_input": int((pd.to_numeric(df["report_count"], errors="coerce").fillna(0) > 1).sum()),
        "records_with_video_evidence": int(rag_df["has_video_evidence"].fillna(False).sum()),
        "records_with_telematics_evidence": int(rag_df["has_telematics_evidence"].fillna(False).sum()),
        "records_with_police_report_evidence": int(rag_df["has_police_report_evidence"].fillna(False).sum()),
    }

    with open(OUTPUT_DIR / "rag_documents_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] wrote", rag_csv)
    print("[OK] wrote", jsonl_path)
    print("[OK] wrote", preview_path)
    print("[OK] wrote", OUTPUT_DIR / "rag_documents_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()