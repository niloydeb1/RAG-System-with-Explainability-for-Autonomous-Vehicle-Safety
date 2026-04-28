# Explainable RAG System

## Setup

1. Clone the repository

```bash
git clone https://github.com/niloydeb1/RAG-System-with-Explainability-for-Autonomous-Vehicle-Safety.git
cd RAG-System-with-Explainability-for-Autonomous-Vehicle-Safety/KGSmile

2. (Optional) Create a virtual environment
 python3 -m venv venv
 source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Create a .env file
nano .env
Add:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

5. Run the application
python frontend.py

6. Open in browser
http://127.0.0.1:7860
