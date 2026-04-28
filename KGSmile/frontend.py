import gradio as gr
import base64
from gpt_rag import ask, parse_hit


def run_query(query):
    try:
        if not query.strip():
            return "Please enter a question.", "", "", ""

        answer, hits, explanation, graph_file = ask(query)

        # format retrieved incidents (same as original style)
        lines = []
        for i, hit in enumerate(hits, start=1):
            hit_id, score, fields = parse_hit(hit)
            title = fields.get("document_title", hit_id)
            lines.append(f"{i}. [{score:.4f}] {title}")

        retrieved_text = "\n".join(lines)

        # embed graph cleanly
        with open(graph_file, "r", encoding="utf-8") as f:
            html = f.read()

        encoded = base64.b64encode(html.encode()).decode()

        graph_html = f"""
        <iframe src="data:text/html;base64,{encoded}"
                width="100%" height="450"
                style="border:none;">
        </iframe>
        """

        return answer, retrieved_text, explanation, graph_html

    except Exception as e:
        return f"Error: {str(e)}", "", "", ""


with gr.Blocks(title="AV Safety RAG") as demo:
    gr.Markdown(
        "## AV Safety Incident RAG System\n"
        "Ask questions about autonomous vehicle safety patterns using NHTSA incident reports."
    )

    # Input
    query_box = gr.Textbox(
        label="Query",
        placeholder='e.g. "Were there pedestrian incidents with injuries?"',
        lines=2,
    )

    submit_btn = gr.Button("Ask", variant="primary")

    # Output sections (clean + structured)
    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)

    retrieved_box = gr.Textbox(
        label="Retrieved Incidents",
        lines=6,
        interactive=False
    )

    explanation_box = gr.Textbox(
        label="Explanation (Top Patterns)",
        lines=6,
        interactive=False
    )

    graph_box = gr.HTML(label="Explanation Graph")

    submit_btn.click(
        fn=run_query,
        inputs=query_box,
        outputs=[answer_box, retrieved_box, explanation_box, graph_box]
    )

    query_box.submit(
        fn=run_query,
        inputs=query_box,
        outputs=[answer_box, retrieved_box, explanation_box, graph_box]
    )

    gr.Examples(
        examples=[
            ["What types of crashes happened most often when the AV was stopped?"],
            ["Were there any pedestrian incidents with injuries?"],
            ["What happened in incidents where automation was not within ODD?"],
        ],
        inputs=query_box,
    )

if __name__ == "__main__":
    demo.launch()
