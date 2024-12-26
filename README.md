# Project Structure:
#
# multimodal_rag/
# ├── core/
# │   ├── __init__.py
# │   ├── document_processor.py
# │   ├── image_processor.py
# │   ├── model_manager.py
# │   ├── retriever.py
# │   ├── exceptions.py
# │   └── utils.py
# ├── outputs/
# │   └── .gitkeep
# ├── .env
# ├── requirements.txt
# ├── main.py
# └── README.md

# requirements.txt
gradio==4.19.2
langchain==0.1.9
langchain-google-genai==0.0.11
faiss-cpu==1.7.4
pypdf==4.0.1
python-magic==0.4.27
Pillow==10.2.0
python-dotenv==1.0.0
spacy==3.7.4
scikit-learn==1.4.1.post1
nltk==3.8.1
PyMuPDF==1.23.26

# .env
GOOGLE_API_KEY=your_api_key_here
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true

# core/exceptions.py


