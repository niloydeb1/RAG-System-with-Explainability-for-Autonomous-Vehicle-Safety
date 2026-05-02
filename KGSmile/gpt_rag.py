import os
import re
import json
from collections import Counter
from dotenv import load_dotenv

# Pinecone & OpenAI Native Clients
from pinecone import Pinecone
from openai import OpenAI
from pyvis.network import Network

# Correct Modern LangChain Integration Imports
from langchain_pinecone import PineconeVectorStore as LangChainPinecone
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "av-safety-rag-index"
NAMESPACE = "nhtsa-av-cases"


# ---------------- Retrieval & LangChain Orchestration ----------------
def retrieve(query):
    """
    Retrieves previous incident cases from the Pinecone vector database
    using LangChain for embedding and orchestration.
    """
    # Force OpenAI's model to output exactly 1024 dimensions to match index schema
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large",
        dimensions=1024
    )

    # Explicitly declares text metadata field to prevent skipping
    vectorstore = LangChainPinecone(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
        text_key="chunk_text"
    )

    # Retrieve top 8 context chunks (optimal for perturbation testing)
    docs = vectorstore.similarity_search(query, k=8)

    # Transform retrieved LangChain documents into the unified hits format
    hits = []
    for d in docs:
        hits.append({
            "_id": d.metadata.get("id", "chunk_id"),
            "_score": d.metadata.get("score", 1.0),
            "fields": {
                "chunk_text": d.page_content,
                "crash_with": d.metadata.get("crash_with", "Unknown"),
                "injury_severity": d.metadata.get("injury_severity", "No Injury")
            }
        })
    return hits


def parse_hit(hit):
    """Safely extracts response metadata."""
    if isinstance(hit, dict):
        return hit.get("_id"), hit.get("_score"), hit.get("fields", {})
    return getattr(hit, "_id", None), getattr(hit, "_score", None), getattr(hit, "fields", {}) or {}


# ---------------- KG-SMILE Scorer Helpers ----------------
def compute_similarity(text1, text2):
    """Computes token-based overlap similarity between the baseline and perturbed answers."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))


def score_hits_by_perturbation(client, query, hits, original_answer):
    """
    True KG-SMILE: Measures the exact semantic impact of each context chunk on the
    LLM's reasoning by masking individual chunks and observing output changes.
    """
    if not hits:
        return []

    scored_hits = []

    # Temporarily remove each hit to calculate its specific attribution
    for i in range(len(hits)):
        masked_hits = [h for j, h in enumerate(hits) if i != j]
        masked_context = "\n".join([parse_hit(h)[2].get("chunk_text", "") for h in masked_hits])

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": masked_context + "\n\nQ: " + query}],
                temperature=0
            )
            masked_answer = response.choices[0].message.content

            # Semantic distance represents context importance
            sim = compute_similarity(original_answer, masked_answer)
            importance_score = 1.0 - sim

            scored_hits.append((importance_score, hits[i]))
        except Exception:
            scored_hits.append((0.0, hits[i]))

    # Sort in descending order of attribution impact
    scored_hits.sort(key=lambda x: x[0], reverse=True)

    # Filter for the most important hits (relevance threshold)
    filtered_hits = [hit for score, hit in scored_hits if score >= 0.05]
    return filtered_hits if filtered_hits else hits


# ---------------- Dynamic Causal Chain Extraction ----------------
def extract_paths_with_llm(client, hits, answer):
    """
    Dynamically extracts logical causal pathways exactly aligned with the
    LLM's synthesis to avoid keyword-matching limitations or visual drift.
    """
    paths = []
    for h in hits:
        _, _, f = parse_hit(h)
        text = f.get("chunk_text", "")

        prompt = f"""
        Based on this AV incident context: "{text}"
        And this specific generated answer: "{answer}"
        Extract the logical causal reasoning chain as 3 to 4 sequential labels separated by arrows (->).
        Example: Pedestrian -> Dark Conditions -> Collision -> Minor Injury
        Keep each node label extremely short (1-3 words max).
        Avoid generic fallbacks. Focus specifically on the conditions or factors discussed.
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            chain_text = res.choices[0].message.content.strip()
            chain = [node.strip() for node in chain_text.split("->") if node.strip()]
            if len(chain) >= 2:
                paths.append(chain)
        except Exception:
            continue
    return paths


def select_top_paths(paths, k=3):
    """Selects the top k most frequent causal pathways."""
    if not paths:
        return []
    counts = Counter(tuple(p) for p in paths)
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [list(p) for p, _ in ranked[:k]]


# ---------------- Layered Knowledge Graph Builder ----------------
def build_graph(paths):
    """Generates a structured visual knowledge graph with dynamic vertical spacing."""
    net = Network(height="500px", width="100%", directed=True)

    def get_color(node_name):
        n_low = node_name.lower()
        #  Critical Actors
        if any(w in n_low for w in ["pedestrian", "cyclist", "motorcyclist", "vehicle", "car"]):
            return "#d62728"  # Red

        #  Environmental Context / Situational Factors
        if any(w in n_low for w in ["crossing", "intersection", "stopped", "dark", "visibility", "rain", "fog", "glare", "weather"]):
            return "#9467bd"  # Purple

        #  Core Incident Events
        if any(w in n_low for w in ["collision", "crash", "strike", "disengagement"]):
            return "#000000"  # Black

        #  Safe Outcomes (Prioritize these checks before general injury)
        if any(w in n_low for w in ["no injury", "no injuries", "zero injuries", "safe", "avoided"]):
            return "#2ca02c"  # Green

        #  Harm / Injury Outcomes
        if "injury" in n_low:
            return "#ff7f0e"  # Orange

        #  Fallback (Gray makes errors easier to diagnose)
        return "#7f7f7f"

    y_offsets = Counter()
    added = set()

    for path in paths:
        path_len = len(path)
        for i in range(path_len - 1):
            src = path[i]
            dst = path[i+1]

            for step_idx, n in enumerate([src, dst]):
                if n not in added:
                    col_idx = min(4, i + step_idx)
                    y_val = y_offsets[col_idx] * 90
                    y_offsets[col_idx] += 1

                    net.add_node(
                        n,
                        label=n,
                        x=col_idx * 240,
                        y=y_val,
                        color=get_color(n),
                        physics=False
                    )
                    added.add(n)

            net.add_edge(src, dst, arrows="to", width=5, color="red")

    net.write_html("graph.html")
    return "graph.html"


# ---------------- Automated RAG Validation Framework ----------------
def evaluate_output(context, answer, paths):
    """
    Automated LLM 'as-a-judge' evaluator based on your project outline.
    Evaluates relevance, correctness, and faithfulness.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    graph_str = "; ".join([" -> ".join(p) for p in paths])

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


# ---------------- Core Pipeline Execution ----------------
def ask(query):
    """Accepts natural language queries, returns grounding evidence, and visualizes logic."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    #  Fetch retrieved incident cases using LangChain Orchestration
    hits = retrieve(query)
    context = "\n".join([parse_hit(h)[2].get("chunk_text", "") for h in hits])

    #  Context-grounded synthesis
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": context + "\n\nQ: " + query}],
        temperature=0
    )
    answer = response.choices[0].message.content

    #  Relevancy Guardrail: Prevent graph drift if context does not discuss query
    no_data_phrases = [
        "no mention", "not found", "does not contain", "no incident",
        "no reports", "are not mentioned", "there is no indication", "does not discuss"
    ]
    if any(phrase in answer.lower() for phrase in no_data_phrases):
        explanation = "The graph is empty because the retrieved context does not contain any relevant instances of your query."
        net = Network(height="500px", width="100%", directed=True)
        net.add_node("No Data Found", label="No Data Found", color="#808080", x=0, y=0, physics=False)
        net.write_html("graph.html")

        evaluation = evaluate_output(context, answer, [])
        return answer, explanation, "graph.html", evaluation

    #  KG-SMILE Perturbation Scorer
    important_hits = score_hits_by_perturbation(client, query, hits, answer)

    #  Dynamic chain extraction
    paths = extract_paths_with_llm(client, important_hits, answer)
    top_paths = select_top_paths(paths, k=3)

    #  Build the PyVis visual output
    graph_file = build_graph(top_paths)

    explanation = "Top Explanation Paths:\n"
    for i, p in enumerate(top_paths):
        explanation += f"{i+1}. {' → '.join(p)}\n"

    #  Evaluate the complete generated result
    evaluation = evaluate_output(context, answer, top_paths)

    return answer, explanation, graph_file, evaluation
