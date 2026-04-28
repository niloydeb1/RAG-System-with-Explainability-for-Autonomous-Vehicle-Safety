import os
import random
from collections import Counter

from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from pyvis.network import Network

# =========================
# SETUP
# =========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NAMESPACE = "nhtsa-av-cases"
INDEX_NAME = "av-safety-rag-index"

DATA_FIELDS = [
    "document_title",
    "chunk_text",
    "state",
    "city",
    "crash_with",
    "injury_severity",
]

SYSTEM_INSTRUCTIONS = """
You analyze patterns in autonomous vehicle safety incidents.
Explain patterns clearly using retrieved cases.
"""

# =========================
# RETRIEVAL
# =========================
def retrieve(pc, query, top_k=10):
    index = pc.Index(INDEX_NAME)

    result = index.search(
        namespace=NAMESPACE,
        query={"inputs": {"text": query}, "top_k": top_k},
        fields=DATA_FIELDS,
    )

    if isinstance(result, dict):
        return result.get("result", {}).get("hits", [])

    result_obj = getattr(result, "result", None)
    return getattr(result_obj, "hits", []) if result_obj else []


def parse_hit(hit):
    if isinstance(hit, dict):
        return hit.get("_id"), hit.get("_score"), hit.get("fields", {})

    return (
        getattr(hit, "_id", None),
        getattr(hit, "_score", None),
        getattr(hit, "fields", {}) or {},
    )


def construct_context(hits):
    parts = []
    for i, hit in enumerate(hits, start=1):
        _, score, fields = parse_hit(hit)
        parts.append(
            f"INCIDENT {i} (score: {score:.4f})\n"
            f"{fields.get('chunk_text', '')}"
        )
    return "\n\n".join(parts)


# =========================
# REASONING
# =========================
def normalize_outcome(injury):
    if not injury:
        return None
    if "Minor" in injury:
        return "Minor"
    if "Moderate" in injury:
        return "Moderate"
    if "Serious" in injury:
        return "Serious"
    return "No Injury"


def extract_event_chain(fields):
    text = (fields.get("chunk_text") or "").lower()

    chain = []

    if fields.get("crash_with") == "Non-Motorist: Pedestrian":
        chain.append("Pedestrian")

    if "crossing" in text:
        chain.append("Crossing")
    elif "intersection" in text:
        chain.append("Intersection")

    if "stopped" in text:
        chain.append("Stopped Vehicle")

    chain.append("Collision")

    outcome = normalize_outcome(fields.get("injury_severity"))
    if outcome:
        chain.append(outcome)

    return chain


def get_top_k_paths(hits, k=3):
    counter = Counter()

    for hit in hits:
        _, _, fields = parse_hit(hit)
        chain = extract_event_chain(fields)

        if len(chain) >= 3:
            counter[tuple(chain)] += 1

    if not counter:
        return ["No strong patterns found."]

    total = sum(counter.values())
    top = counter.most_common(k)

    results = []
    for path, count in top:
        conf = round((count / total) * 100, 1)
        results.append(f"{' → '.join(path)} ({conf}%)")

    return results


# =========================
# GRAPH (CLEAN VERSION)
# =========================
def build_visual_graph(hits):
    net = Network(height="500px", width="100%", directed=True)

    edge_weights = Counter()
    path_counts = Counter()

    for hit in hits:
        _, _, fields = parse_hit(hit)
        chain = extract_event_chain(fields)

        if len(chain) >= 3:
            path_counts[tuple(chain)] += 1
            for i in range(len(chain) - 1):
                edge_weights[(chain[i], chain[i+1])] += 1

    if not edge_weights:
        net.add_node("No Data")
        net.write_html("graph.html")
        return "graph.html"

    top_paths = [p for p, _ in path_counts.most_common(3)]

    allowed_edges = set()
    for path in top_paths:
        for i in range(len(path) - 1):
            allowed_edges.add((path[i], path[i+1]))

    def relation(src, dst):
        if src == "Pedestrian":
            return "involved_in"
        if dst in ["Crossing", "Intersection"]:
            return "occurs_at"
        if dst == "Stopped Vehicle":
            return "vehicle_state"
        if dst == "Collision":
            return "results_in"
        if dst in ["Minor", "Moderate", "Serious", "No Injury"]:
            return "outcome"
        return "leads_to"

    # Left-to-right layout
    STAGE_X = {
        "Pedestrian": 0,
        "Crossing": 1,
        "Intersection": 1,
        "Stopped Vehicle": 2,
        "Collision": 3,
        "Minor": 4,
        "Moderate": 4,
        "Serious": 4,
        "No Injury": 4,
    }

    outcome_y = {
        "Minor": -100,
        "Moderate": 0,
        "Serious": 100,
        "No Injury": 200,
    }

    def node_x(n): return STAGE_X.get(n, 2) * 250
    def node_y(n):
        if n in outcome_y:
            return outcome_y[n]
        return random.randint(-150, 150)

    max_w = max(edge_weights.values())

    added = set()

    # Draw edges
    for (src, dst), w in edge_weights.items():
        if (src, dst) not in allowed_edges:
            continue

        conf = w / max_w

        if src not in added:
            net.add_node(src, label=src, x=node_x(src), y=node_y(src), physics=False)
            added.add(src)

        if dst not in added:
            net.add_node(dst, label=dst, x=node_x(dst), y=node_y(dst), physics=False)
            added.add(dst)

        net.add_edge(
            src,
            dst,
            label=f"{conf:.2f}",
            title=relation(src, dst),
            width=2 + 5 * conf,
            smooth={"type": "curvedCW", "roundness": 0.2}
        )

    # Highlight top paths
    colors = ["red", "orange", "green"]

    for idx, path in enumerate(top_paths):
        nodes = list(path)
        for i in range(len(nodes) - 1):
            net.add_edge(
                nodes[i],
                nodes[i+1],
                color=colors[idx],
                width=8,
                arrows="to"
            )

    net.set_options("""
    {
      "edges": { "font": { "size": 10 } },
      "nodes": { "font": { "size": 16 } }
    }
    """)

    net.write_html("graph.html")
    return "graph.html"


# =========================
# MAIN
# =========================
def ask(query):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)

    hits = retrieve(pc, query)
    context = construct_context(hits)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content
    top_paths = get_top_k_paths(hits)
    graph_file = build_visual_graph(hits)

    explanation = "Top Explanation Paths:\n" + "\n".join(
        [f"{i+1}. {p}" for i, p in enumerate(top_paths)]
    )

    return answer, hits, explanation, graph_file
