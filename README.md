This project is for Testing TinyLlama/TinyLlama-1.1B-Chat-v1.0 Generative AI in Local Computer.

mkdir .venv
python -m venv .venv
 .\.venv\Scripts\Activate.ps1  for Windows



Need Python 3 and install
pip install transformers torch accelerate fastapi pydantic uvicorn sentence_transformers chromadb
pip install fastapi uvicorn torch transformers sentence-transformers chromadb ddgs curl_cffi
pip install requests beautifulsoup4 lxml




To Run Fast API
uvicorn main:app --reload
