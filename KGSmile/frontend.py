import gradio as gr
from gpt_rag import ask, parse_hit
def run_query(query: str):
    import base64

    try:
        # handle empty input
        if not query.strip():
            return "Please enter a question.", "", "", ""

        # call backend
        answer, hits, paths_text, graph_file = ask(query, verbose=False)

        # format retrieved incidents
        lines = []
        for i, hit in enumerate(hits, start=1):
            hit_id, score, fields = parse_hit(hit)
            title = fields.get("document_title", hit_id)
            lines.append(f"{i}. [{score:.4f}]  {title}")
        retrieved_text = "\n".join(lines)

        # read graph HTML
        with open(graph_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # encode as base64 so it renders in Gradio
        encoded = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")

        iframe_html = f"""
        <iframe src="data:text/html;base64,{encoded}"
                width="100%"
                height="500"
                style="border:none;">
        </iframe>
        """

        return answer, retrieved_text, paths_text, iframe_html

    except Exception as e:
        import traceback
        traceback.print_exc()

        # return error safely so UI doesn't crash
        return f"Error: {str(e)}", "", "", ""
with gr.Blocks(title="AV Safety RAG") as demo:
    gr.Markdown(
        "## AV Safety Incident RAG System\n"
        "Ask questions about autonomous vehicle safety patterns using NHTSA incident reports."
    )

    query_box = gr.Textbox(
        label="Query",
        placeholder='e.g. "What are common factors in rear-end crashes involving AVs?"',
        lines=2,
    )
    submit_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=12, interactive=False)
    retrieved_box = gr.Textbox(label="Retrieved Incidents", lines=6, interactive=False)
    paths_box = gr.Textbox(label="Reasoning Paths", lines=8, interactive=False)
    graph_box = gr.HTML(label="Graph Visualization")

    #graph_box = gr.Textbox(label="Graph Explanation", lines=10, interactive=False)

    submit_btn.click(fn=run_query, inputs=query_box, outputs=[answer_box, retrieved_box, paths_box, graph_box])
    query_box.submit(fn=run_query, inputs=query_box, outputs=[answer_box, retrieved_box, paths_box, graph_box])

    gr.Examples(
        examples=[
            ["What types of crashes happened most often when the AV was stopped?"],
            ["Were there any pedestrian incidents with injuries?"],
            ["What happened in incidents where automation was not within ODD?"],
            ["What should an AV do when approaching a yellow light at high speed?"],
        ],
        inputs=query_box,
    )


if __name__ == "__main__":
    #demo.launch()
    demo.launch(allowed_paths=["."])
