import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
namespace = "nhtsa-av-cases"
INDEX_NAME = "av-safety-rag-index"
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
    #index = pc.Index(index_name)
    index = pc.Index("av-safety-rag-index")
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

    #answer = response.choices[0].message.content
    #return answer, hits
    #answer = response.choices[0].message.content

    # build graph
    #graph = build_graph_from_hits(hits)
    #graph_text = graph_to_text(graph)

    #return answer, hits, graph_text
    answer = response.choices[0].message.content

    paths = extract_reasoning_paths(hits)
    paths_text = "\n".join(paths)

    graph_file = build_visual_graph(hits)
    top_path = get_top_reasoning_path(hits)

    #return answer, hits, paths_text, graph_file
    return answer, hits, paths_text + f"\n\nTOP PATH:\n{top_path}", graph_file

def extract_entities_from_hit(fields):
    return [
        fields.get("crash_with"),
        fields.get("injury_severity"),
        fields.get("within_odd"),
        fields.get("automation_type_engaged"),
        fields.get("state"),
        fields.get("city"),
    ]

from collections import defaultdict

def build_graph_from_hits(hits):
    graph = defaultdict(set)

    for hit in hits:
        _, _, fields = parse_hit(hit)
        entities = extract_entities_from_hit(fields)

        # remove None values
        entities = [e for e in entities if e]

        # connect all entities in the same incident
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                graph[entities[i]].add(entities[j])
                graph[entities[j]].add(entities[i])

    return graph
def graph_to_text(graph):
    lines = []
    for node, neighbors in graph.items():
        if neighbors:
            lines.append(f"{node} → {', '.join(list(neighbors)[:3])}")
    return "\n".join(lines[:10])

def extract_reasoning_paths(hits, max_paths=5):
    paths = []

    for hit in hits:
        _, _, fields = parse_hit(hit)
        chain = extract_event_chain(fields)

        if len(chain) >= 3:
            paths.append(" → ".join(chain))

    return paths[:max_paths]
def get_color(node):
    if "Pedestrian" in node:
        return "red"
    elif "Injury" in node or "Minor" in node or "Serious" in node:
        return "orange"
    elif node in ["Intersection", "Crossing", "Stopped Vehicle"]:
        return "purple"
    elif node == "Collision":
        return "black"
    else:
        return "blue"


def get_size(node):
    if node == "Collision":
        return 35
    elif "Pedestrian" in node:
        return 30
    elif "Injury" in node or "Minor" in node or "Serious" in node:
        return 25
    else:
        return 15

from pyvis.network import Network
import re

def extract_event_chain(fields):
    text = (fields.get("chunk_text") or "").lower()

    chain = []

    # Stage 1: actor
    if fields.get("crash_with") == "Non-Motorist: Pedestrian":
        chain.append("Pedestrian")

    # Stage 2: context (ONLY ONE)
    if "crossing" in text:
        chain.append("Crossing")
    elif "intersection" in text:
        chain.append("Intersection")

    # Stage 3: vehicle state
    if "stopped" in text:
        chain.append("Stopped Vehicle")

    # Stage 4: event (force one direction)
    chain.append("Collision")

    # Stage 5: outcome
    injury = fields.get("injury_severity")
    if injury:
        chain.append(injury)

    return chain


from pyvis.network import Network
from collections import Counter

from pyvis.network import Network
from collections import Counter


def build_visual_graph(hits):
    net = Network(height="500px", width="100%", directed=True)

    edge_weights = Counter()
    path_counts = Counter()

    # -------- Step 1: Build event chains --------
    chains = []
    for hit in hits:
        _, _, fields = parse_hit(hit)
        chain = extract_event_chain(fields)

        if len(chain) >= 3:
            chains.append(chain)
            path_counts[tuple(chain)] += 1

            for i in range(len(chain) - 1):
                edge_weights[(chain[i], chain[i+1])] += 1

    # -------- Step 2: Handle empty case --------
    if not chains:
        net.add_node("No Pattern Found", color="gray", size=20)
        file_path = "graph.html"
        net.write_html(file_path, notebook=False, open_browser=False)
        return file_path

    # -------- Step 3: Find top reasoning path --------
    top_path = path_counts.most_common(1)[0][0]

    # -------- Step 4: Styling helpers --------
    def get_color(node):
        if "Pedestrian" in node:
            return "red"
        elif node == "Collision":
            return "black"
        elif "Injury" in node or "Minor" in node or "Serious" in node:
            return "orange"
        elif node in ["Crossing", "Intersection", "Stopped Vehicle"]:
            return "purple"
        else:
            return "blue"

    def get_size(node):
        if node == "Collision":
            return 40
        elif "Pedestrian" in node:
            return 30
        elif node in top_path:
            return 25
        else:
            return 15

    bad_nodes = {"Other, see Narrative", "Unknown", None}

    # -------- Step 5: Add normal edges --------
    for (src, dst), weight in edge_weights.items():

        if src in bad_nodes or dst in bad_nodes:
            continue

        # prevent reverse loops
        if (dst, src) in edge_weights:
            continue

        net.add_node(src, label=src, color=get_color(src), size=get_size(src))
        net.add_node(dst, label=dst, color=get_color(dst), size=get_size(dst))

        net.add_edge(src, dst, width=1 + weight)

    # -------- Step 6: Highlight top reasoning path --------
    seen_edges = set()
    top_nodes = list(top_path)

    for i in range(len(top_nodes) - 1):
        edge = (top_nodes[i], top_nodes[i+1])

        if edge not in seen_edges:
            net.add_edge(
                edge[0],
                edge[1],
                width=8,
                color="red"
            )
            seen_edges.add(edge)

    # -------- Step 7: Layout --------
    net.force_atlas_2based()

    # -------- Step 8: Save graph --------
    file_path = "graph.html"
    net.write_html(file_path, notebook=False, open_browser=False)

    return file_path

def get_top_reasoning_path(hits):
    from collections import Counter

    path_counts = Counter()

    for hit in hits:
        _, _, fields = parse_hit(hit)
        chain = extract_event_chain(fields)

        if len(chain) >= 3:
            path = tuple(chain)
            path_counts[path] += 1

    if not path_counts:
        return "No strong pattern found."

    top_path = path_counts.most_common(1)[0][0]
    return " → ".join(top_path)
#from pyvis.network import Network


#def build_visual_graph(hits):
 #   net = Network(height="500px", width="100%", directed=True)

  #  for hit in hits:
   #     _, _, fields = parse_hit(hit)

    #    crash = fields.get("crash_with")
     #   city = fields.get("city")
      #  injury = fields.get("injury_severity")

       # nodes = [crash, city, injury]
        #nodes = [n for n in nodes if n]

        #for i in range(len(nodes) - 1):
         #   net.add_node(nodes[i], label=nodes[i])
          #  net.add_node(nodes[i+1], label=nodes[i+1])
           # net.add_edge(nodes[i], nodes[i+1])

    ##file_path = "graph.html"
    #import os

    #file_path = os.path.abspath("graph.html")
    ##net.save_graph(file_path)
    #net.write_html(file_path, notebook=False, open_browser=False)
    #return file_path
