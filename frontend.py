import gradio as gr
from gpt_rag import ask, parse_hit


def run_query(query: str):
    if not query.strip():
        return "Please enter a question.", ""

    answer, hits = ask(query, verbose=False)

    lines = []
    for i, hit in enumerate(hits, start=1):
        hit_id, score, fields = parse_hit(hit)
        title = fields.get("document_title", hit_id)
        lines.append(f"{i}. [{score:.4f}]  {title}")
    retrieved_text = "\n".join(lines)

    return answer, retrieved_text


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

    submit_btn.click(fn=run_query, inputs=query_box, outputs=[answer_box, retrieved_box])
    query_box.submit(fn=run_query, inputs=query_box, outputs=[answer_box, retrieved_box])

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
    demo.launch()
