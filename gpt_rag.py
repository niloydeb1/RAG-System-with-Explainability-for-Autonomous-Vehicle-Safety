import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

PINECONE_API_KEY = "ENTER API CODE"
OPENAI_API_KEY = "ENTER API CODE"
namespace = "nhtsa-av-cases"

data_fields = [
    "document_title",
    "chunk_text",
    "state",
    "city",
    "crash_with",
    "injury_severity",
    "within_odd",
    "automation_type_engaged",
    "primary_reporting_entity",
    "incident_ym",
]

system_instructions = """
You are an expert at analyzing behavior patterns in autonomous vehicle safety incidents using NHTSA Standing General Order reports.
Your goal is to identify observable patterns and contributing factors across the provided incident records, such as common crash types, conditions, locations, or automation states, and use those patterns to answer a question proposed by the user.

Guidelines:
- Ground every claim in the retrieved incident records. Cite the incident case ID in parentheses for each specific claim.
- For pattern or statistical questions: identify what the records have in common (e.g., shared crash type, ODD status, injury outcome) and summarize the pattern clearly.
- For case-based questions: describe the relevant incidents and highlight shared or notable factors.
- For hypothetical or decision-support questions: reason from what similar past incidents show, and be explicit that your answer is based on observed patterns, not AV internal logic.
- If the retrieved records are insufficient to fully answer the question, say so clearly and describe what type of additional data would be needed.
"""


def retrieve(pc: Pinecone, query: str, top_k: int = 10):
    index = pc.Index(index_name)
    result = index.search(
        namespace=namespace,
        query={"inputs": {"text": query}, "top_k": top_k},
        fields=data_fields,
    )

    if isinstance(result, dict):
        return result.get("result", {}).get("hits", [])

    result_obj = getattr(result, "result", None)
    return getattr(result_obj, "hits", []) if result_obj is not None else []


def parse_hit(hit):
    if isinstance(hit, dict):
        return hit.get("_id"), hit.get("_score"), hit.get("fields", {})
    hit_id = getattr(hit, "_id", None)
    score = getattr(hit, "_score", None)
    fields = getattr(hit, "fields", {}) or {}
    return hit_id, score, fields


def construct_context(hits) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        hit_id, score, fields = parse_hit(hit)
        parts.append(
            f"INCIDENT {i} (ID: {hit_id}, relevance score: {score:.4f})\n"
            f"{fields.get('chunk_text', '')}"
        )
    return "\n\n".join(parts)


def ask(query: str, hits=None, verbose: bool = True):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)

    if hits is None:
        hits = retrieve(pc, query)
    context = construct_context(hits)

    if verbose:
        print(f"\nRetrieved {len(hits)} incidents for: {query!r}")
        for i, hit in enumerate(hits, start=1):
            hit_id, score, fields = parse_hit(hit)
            print(f"  {i}. {fields.get('document_title', hit_id)}  (score={score:.4f})")

    messages = [
        {"role": "system", "content": system_instructions},
        {
            "role": "user",
            "content": (
                f"Relevant incident records:\n\n{context}\n\n"
                f"Question: {query}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content
    return answer, hits


