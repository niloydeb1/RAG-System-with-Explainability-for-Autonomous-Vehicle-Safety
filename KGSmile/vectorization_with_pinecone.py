import json
import math
import time
from pathlib import Path

from pinecone import Pinecone

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# USER SETTINGS
# =========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print("Loaded Pinecone key:", PINECONE_API_KEY)
INDEX_NAME = "av-safety-rag-index"
NAMESPACE = "nhtsa-av-cases"
JSONL_PATH = Path(r"Data/clean_reports_outputs/merged_outputs/incident_cases_outputs/rag_documents_outputs/pinecone_integrated_records.jsonl")

# One-time clean restart:
# True  -> delete existing index and rebuild from scratch
# False -> keep existing index
RESET_INDEX = False

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "llama-text-embed-v2"
TEXT_FIELD = "chunk_text"
METRIC = "cosine"

# Conservative settings to stay below Starter limits
BATCH_SIZE = 12                # well below 96 text-record limit
SAFE_TPM_BUDGET = 120_000      # well below 250k tokens/minute
WINDOW_SECONDS = 60

# =========================
# HELPERS
# =========================
def get_status_ready(desc) -> bool:
    status = getattr(desc, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready", False))
    if status is not None and hasattr(status, "ready"):
        return bool(status.ready)
    if isinstance(desc, dict):
        return bool(desc.get("status", {}).get("ready", False))
    return False


def wait_until_index_ready(pc: Pinecone, timeout_seconds: int = 900):
    start = time.time()
    while True:
        desc = pc.describe_index(name=INDEX_NAME)
        if get_status_ready(desc):
            print("[OK] Index is ready.")
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for index to become ready.")
        print("[INFO] Waiting for index to become ready...")
        time.sleep(10)


def wait_until_index_deleted(pc: Pinecone, timeout_seconds: int = 900):
    start = time.time()
    while True:
        if not pc.has_index(INDEX_NAME):
            print("[OK] Index deleted.")
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for index deletion.")
        print("[INFO] Waiting for index deletion...")
        time.sleep(10)


def recreate_index(pc: Pinecone):
    if pc.has_index(INDEX_NAME) and RESET_INDEX:
        print(f"[INFO] Deleting existing index: {INDEX_NAME}")
        pc.delete_index(INDEX_NAME)
        wait_until_index_deleted(pc)

    if not pc.has_index(INDEX_NAME):
        print(f"[INFO] Creating index: {INDEX_NAME}")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud=CLOUD,
            region=REGION,
            embed={
                "model": EMBED_MODEL,
                "metric": METRIC,
                "field_map": {"text": TEXT_FIELD},
            },
            deletion_protection="disabled",
            tags={
                "project": "av-safety-rag",
                "dataset": "nhtsa-sgo",
                "stage": "prototype",
            },
        )
        print("[OK] Index creation requested.")
        wait_until_index_ready(pc)
    else:
        print(f"[OK] Index already exists: {INDEX_NAME}")
        wait_until_index_ready(pc)


def sanitize_record_for_pinecone(record: dict) -> dict:
    """
    Keep only Pinecone-supported metadata types:
    string, number, boolean, or list of strings.
    Drop None/null and empty strings.
    """
    cleaned = {}

    for key, value in record.items():
        if key == "_id":
            cleaned[key] = str(value)
            continue

        if value is None:
            continue

        if isinstance(value, float) and math.isnan(value):
            continue

        if isinstance(value, str):
            value = value.strip()
            if value:
                cleaned[key] = value
            continue

        if isinstance(value, bool):
            cleaned[key] = value
            continue

        if isinstance(value, (int, float)):
            cleaned[key] = value
            continue

        if isinstance(value, list):
            out = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    item = item.strip()
                    if item:
                        out.append(item)
                else:
                    item = str(item).strip()
                    if item:
                        out.append(item)
            if out:
                cleaned[key] = out
            continue

        fallback = str(value).strip()
        if fallback:
            cleaned[key] = fallback

    return cleaned


def load_records(jsonl_path: Path):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Could not find file: {jsonl_path}")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            if "_id" not in rec:
                raise ValueError(f"Missing _id on line {line_num}")
            if TEXT_FIELD not in rec:
                raise ValueError(f"Missing {TEXT_FIELD!r} on line {line_num}")

            rec = sanitize_record_for_pinecone(rec)

            if "_id" not in rec:
                raise ValueError(f"_id missing after sanitization on line {line_num}")
            if TEXT_FIELD not in rec:
                raise ValueError(f"{TEXT_FIELD!r} missing after sanitization on line {line_num}")

            records.append(rec)

    return records


def estimate_tokens_for_record(record: dict) -> int:
    """
    Rough estimate: ~4 characters per token for English text.
    Good enough for upload pacing.
    """
    text = record.get(TEXT_FIELD, "")
    if not isinstance(text, str):
        text = str(text)
    return max(1, len(text) // 4)


def upsert_records_with_throttle(pc: Pinecone, records):
    index = pc.Index(INDEX_NAME)
    total = len(records)
    print(f"[INFO] Uploading {total} records to namespace: {NAMESPACE}")

    window_start = time.time()
    tokens_used = 0

    for start in range(0, total, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        batch_token_est = sum(estimate_tokens_for_record(r) for r in batch)

        # Reset the local minute window if needed
        elapsed = time.time() - window_start
        if elapsed >= WINDOW_SECONDS:
            window_start = time.time()
            tokens_used = 0
            elapsed = 0

        # Stay below the safe token budget
        if tokens_used + batch_token_est > SAFE_TPM_BUDGET:
            sleep_for = max(0, WINDOW_SECONDS - elapsed + 2)
            print(f"[INFO] Sleeping {sleep_for:.1f}s to stay under token/minute budget...")
            time.sleep(sleep_for)
            window_start = time.time()
            tokens_used = 0

        retries = 0
        while True:
            try:
                index.upsert_records(NAMESPACE, batch)
                tokens_used += batch_token_est
                end = start + len(batch)
                print(
                    f"[OK] Upserted records {start + 1} to {end} | "
                    f"batch_est_tokens={batch_token_est} | "
                    f"window_est_tokens={tokens_used}"
                )
                break

            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = (
                        "429" in msg
                        or "resource_exhausted" in msg
                        or "too many requests" in msg
                        or "max tokens per minute" in msg
                )

                if not is_rate_limit:
                    raise

                retries += 1
                wait_seconds = min(180, 20 * (2 ** (retries - 1)))
                print(f"[WARN] Rate-limited. Sleeping {wait_seconds}s, then retrying this batch...")
                time.sleep(wait_seconds)

                # Reset local pacing window after a hard throttle
                window_start = time.time()
                tokens_used = 0

                if retries >= 6:
                    raise RuntimeError(
                        f"Too many rate-limit retries on batch starting at record {start + 1}"
                    ) from e

    print("[OK] All records uploaded.")


def run_test_queries(pc: Pinecone):
    index = pc.Index(INDEX_NAME)

    test_queries = [
        "rear-end crashes where the autonomous vehicle was stopped",
        "pedestrian crossing incidents",
        "incidents within ODD with injuries",
        "left turn conflicts at intersections",
    ]

    fields_to_return = [
        "document_title",
        "chunk_text",
        "state",
        "city",
        "crash_with",
        "injury_severity",
        "within_odd",
        "automation_type_engaged",
        "primary_reporting_entity",
    ]

    for q in test_queries:
        print("\n" + "=" * 90)
        print("QUERY:", q)
        result = index.search(
            namespace=NAMESPACE,
            query={
                "inputs": {"text": q},
                "top_k": 3,
            },
            fields=fields_to_return,
        )

        if isinstance(result, dict):
            hits = result.get("result", {}).get("hits", [])
        else:
            result_obj = getattr(result, "result", None)
            hits = getattr(result_obj, "hits", []) if result_obj is not None else []

        for i, hit in enumerate(hits, start=1):
            if isinstance(hit, dict):
                hit_id = hit.get("_id")
                hit_score = hit.get("_score")
                fields = hit.get("fields", {})
            else:
                hit_id = getattr(hit, "_id", None)
                hit_score = getattr(hit, "_score", None)
                fields = getattr(hit, "fields", {}) or {}

            print(f"\nResult {i}")
            print("ID:", hit_id)
            print("Score:", hit_score)
            print("Title:", fields.get("document_title"))
            print("State:", fields.get("state"))
            print("City:", fields.get("city"))
            print("Crash With:", fields.get("crash_with"))
            print("Injury Severity:", fields.get("injury_severity"))
            print("Within ODD:", fields.get("within_odd"))
            print("Automation:", fields.get("automation_type_engaged"))
            snippet = fields.get("chunk_text", "")
            print("Snippet:")
            print(snippet[:700] + ("..." if len(snippet) > 700 else ""))


def main():
    if not PINECONE_API_KEY:
        raise EnvironmentError("API_KEY is empty.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    recreate_index(pc)

    records = load_records(JSONL_PATH)
    print(f"[INFO] Loaded {len(records)} sanitized records.")

    upsert_records_with_throttle(pc, records)

    print("[INFO] Waiting briefly before test search...")
    time.sleep(10)

    run_test_queries(pc)


if __name__ == "__main__":
    main()
