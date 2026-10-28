# Career Chatbot

A Streamlit chatbot that answers career-guidance questions from a curated question-and-answer dataset.

## Why this exists
Practice with retrieval-augmented generation (RAG): grounding an LLM's answers in a small, known dataset instead of its general knowledge.

## What it does
- Loads `data.csv` (career questions and answers) and embeds it with BGE embeddings
- Stores the vectors in a local FAISS index ("Create Knowledgebase" button)
- Retrieves the closest entries for a question and answers with GPT-4 through a `RetrievalQA` chain
- Falls back to general knowledge when the data has no match, and declines questions unrelated to careers

## Project structure
```
careerchatbot/
├── .devcontainer/       Dev container for Codespaces
├── faiss_index/         Saved FAISS vector index
├── data.csv             Career Q&A source data
├── langchain_helper.py  Index building and QA chain
├── main.py              Streamlit app
└── requirements.txt     Python dependencies
```

## Getting started
Requirements: Python 3.10+, an OpenAI API key.
```bash
git clone https://github.com/abhijha8287/careerchatbot.git
cd careerchatbot
pip install -r requirements.txt
mkdir -p .streamlit && echo 'api_key = "sk-..."' > .streamlit/secrets.toml
streamlit run main.py
```
Click "Create Knowledgebase" once to rebuild the index, then ask a question.

## Configuration
| Variable | Purpose |
|:--|:--|
| `api_key` in `.streamlit/secrets.toml` | OpenAI API key |

## Tech stack
Python, LangChain, OpenAI GPT-4, HuggingFace BGE embeddings, FAISS, Streamlit

## Notes
Live demo: https://careerchatbotbyabhishek.streamlit.app (on the free Streamlit tier it may take a few seconds to wake up).
