import os
import re
import json
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from pyvis.network import Network

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Neo4j Retrieval Phase ----------------
def retrieve_subgraph(query_text):
    """Retrieves exact node-edge-node triples from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Extracts keywords to query the graph
    keywords = [w.strip().lower() for w in re.findall(r'\b\w{4,}\b', query_text)]

    cypher_query = """
    MATCH (n)-[r]->(m)
    WHERE any(k IN $keywords WHERE toLower(n.name) CONTAINS k OR toLower(m.name) CONTAINS k OR toLower(type(r)) CONTAINS k)
    RETURN n.name AS source, type(r) AS relationship, m.name AS target
    LIMIT 12
    """

    triples = []
    try:
        with driver.session() as session:
            result = session.run(cypher_query, keywords=keywords)
            for record in result:
                triples.append({
                    "source": record["source"],
                    "relationship": record["relationship"],
                    "target": record["target"]
                })
    except Exception as e:
        print(f"[Neo4j Error]: {e}")
        # Fallback data if Neo4j is not connected
        triples = [
            {"source": "Software Bug", "relationship": "CAUSES", "target": "Disengagement"},
            {"source": "Disengagement", "relationship": "LEADS_TO", "target": "Collision"},
            {"source": "Dark Conditions", "relationship": "EXACERBATES", "target": "Detection Latency"}
        ]
    finally:
        driver.close()

    return triples


# ---------------- LLM Helper Utilities ----------------
def compute_similarity(text1, text2):
    """Computes token-based overlap similarity between baseline and perturbed answers."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))


def generate_answer(triples, query):
    """Generates an answer using the graph triples as context."""
    context_str = "\n".join([f"({t['source']})-[{t['relationship']}]->({t['target']})" for t in triples])
    prompt = f"Context from Knowledge Graph:\n{context_str}\n\nQuestion: {query}\nProvide a concise answer based on this context."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# ---------------- KG-SMILE Perturbation Scorer ----------------
def score_triples_by_perturbation(query, triples, original_answer):
    """Scores graph components by masking them and evaluating output divergence."""
    if not triples:
        return []

    scored_triples = []
    for i, target_triple in enumerate(triples):
        masked_triples = [t for j, t in enumerate(triples) if i != j]
        try:
            masked_answer = generate_answer(masked_triples, query)
            sim = compute_similarity(original_answer, masked_answer)
            importance_score = 1.0 - sim
            scored_triples.append((importance_score, target_triple))
        except Exception:
            scored_triples.append((0.0, target_triple))

    scored_triples.sort(key=lambda x: x[0], reverse=True)
    return scored_triples


# ---------------- Graph Visualization Matching ----------------
def build_graph(scored_triples):
    """Generates a stable, beautifully spaced network graph with readable node labels."""
    net = Network(height="500px", width="100%", directed=True)

    # Use robust Barnes-Hut physics for maximum spacing and explicit overlap avoidance
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.1,
          "springLength": 220,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 1
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        }
      }
    }
    """)

    def get_color(node_name):
        n_low = node_name.lower()
        if any(w in n_low for w in ["pedestrian", "cyclist", "motorcyclist", "vehicle", "car"]):
            return "#d62728"  # Red
        if any(w in n_low for w in ["crossing", "intersection", "stopped", "dark", "visibility", "rain", "fog", "glare", "weather"]):
            return "#9467bd"  # Purple
        if any(w in n_low for w in ["collision", "crash", "strike", "disengagement"]):
            return "#000000"  # Black
        if any(w in n_low for w in ["no injury", "no injuries", "zero injuries", "safe", "avoided"]):
            return "#2ca02c"  # Green
        if "injury" in n_low:
            return "#ff7f0e"  # Orange
        return "#7f7f7f"     # Gray

    added = set()

    for score, t in scored_triples:
        src, rel, dst = t["source"], t["relationship"], t["target"]

        for n in [src, dst]:
            if n not in added:
                net.add_node(
                    n,
                    label=n,
                    shape="box",  # Wraps label inside a container to prevent label overlap
                    color={"background": "#ffffff", "border": get_color(n)},
                    borderWidth=2,
                    font={"size": 14, "color": "#2c3e50", "face": "arial"},
                    margin=10
                )
                added.add(n)

        # Map perturbation impact to edge thickness
        thickness = int(2 + (score * 12))
        edge_color = "#d62728" if score >= 0.05 else "#b0b0b0"

        net.add_edge(
            src,
            dst,
            label=rel,
            arrows="to",
            width=thickness,
            color=edge_color,
            font={"size": 11, "align": "top"}
        )

    # Save out the graph html
    net.write_html("graph.html")
    return "graph.html"

# ---------------- Exact Restored Evaluation Framework ----------------
def evaluate_output(context, answer, paths):
    """The original 3-metric LLM evaluation (Faithfulness, Fidelity, Completeness)."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    graph_str = "; ".join([f"{t['source']} -> {t['target']}" for t in paths])

    prompt = f"""
    You are an expert safety auditor evaluating an Explainable GraphRAG system for AV safety.
    Analyze the generated components to grade their overall relevance and correctness.

    [Context from Database]:
    {context}

    [Generated Answer]:
    {answer}

    [Extracted Pathways in Graph]:
    {graph_str}

    Evaluate and rate the following 3 metrics on a scale of 0.0 to 1.0 (with brief reasoning):
    1. Faithfulness: Is the answer fully grounded in the retrieved context without hallucinations?
    2. Fidelity: Does the graph accurately summarize the reasoning in the answer?
    3. Completeness: Are all critical entities (e.g., weather, actors) properly represented in the graph?
    Return your grading response in structured JSON format.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.dumps(json.loads(res.choices[0].message.content), indent=2)
    except Exception as e:
        return json.dumps({"error": f"Evaluation failed: {str(e)}"})


# ---------------- Core Pipeline Orchestration ----------------
def ask(query):
    """Accepts queries, returns grounding evidence, and evaluates exactly as before."""
    #  Retrieve explicitly from Neo4j
    triples = retrieve_subgraph(query)

    if not triples:
        explanation = "The graph is empty because the retrieved context does not contain any relevant instances of your query."
        net = Network(height="500px", width="100%", directed=True)
        net.add_node("No Data Found", label="No Data Found", color="#808080", x=0, y=0, physics=False)
        net.write_html("graph.html")
        evaluation = evaluate_output("", "No information found.", [])
        return "No data found.", explanation, "graph.html", evaluation

    #  Reconstruct original string representation of context for evaluation
    context = "\n".join([f"({t['source']})-[{t['relationship']}]->({t['target']})" for t in triples])

    #  Context-grounded synthesis
    answer = generate_answer(triples, query)

    #  Perturbation Scoring
    scored_triples = score_triples_by_perturbation(query, triples, answer)

    # Restored original k=3 path filtering for text explanation
    top_triples = [t for score, t in scored_triples[:3]]

    #  Build visualization with original colors
    graph_file = build_graph(scored_triples)

    explanation = "Top Explanation Paths:\n"
    for i, t in enumerate(top_triples):
        explanation += f"{i+1}. {t['source']} → {t['relationship']} → {t['target']}\n"

    #  Apply restored evaluation framework
    evaluation = evaluate_output(context, answer, top_triples)

    return answer, explanation, graph_file, evaluation
