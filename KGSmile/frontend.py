import gradio as gr
from gpt_rag import ask

def run_query(query):
    """
    Calls the gpt_rag pipeline, unpacks exactly 4 items,
    and returns text/HTML content to the Gradio interface.
    """
    # Unpack the exact 4 elements returned by ask()
    answer, explanation, graph_file, evaluation = ask(query)

    # Read the PyVis file content safely
    with open(graph_file, "r", encoding="utf-8") as f:
        graph_html_content = f.read()

    # Wrap the full HTML content in an iframe so the browser renders it perfectly
    iframe_html = f'''
    <iframe srcdoc="{graph_html_content.replace('"', '&quot;')}"
            width="100%"
            height="550px"
            style="border:none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    </iframe>
    '''

    return answer, explanation, iframe_html, evaluation

# Build and launch the frontend
def launch_frontend():
    with gr.Blocks(title="Explainable GraphRAG for AV Safety") as demo:
        gr.Markdown("# Explainable GraphRAG System for Autonomous Vehicle Safety")
        gr.Markdown("Query previous AV incidents and explore dynamic causal chains extracted via the KG-SMILE framework.")

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Natural Language Query",
                    placeholder="e.g., How do vehicle software disengagements contribute to crashes?",
                    lines=2
                )
                submit_btn = gr.Button("Analyze Scenario", variant="primary")

            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Generated Synthesis / Answer", lines=6)
                explanation_output = gr.Textbox(label="Top Extracted Graph Paths", lines=4)
                eval_output = gr.Code(label="Automated LLM Evaluation Grading (JSON)", language="json")

        with gr.Row():
            with gr.Column():
                graph_output = gr.HTML(label="KG-SMILE Knowledge Graph Visualization")

        submit_btn.click(
            fn=run_query,
            inputs=[query_input],
            outputs=[answer_output, explanation_output, graph_output, eval_output]
        )

    demo.launch()

if __name__ == "__main__":
    launch_frontend()
