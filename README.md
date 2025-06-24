# RAG-forAITest


## RAG Cybersecurity Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers cybersecurity-related questions using a combination of document retrieval and a local large language model (**Mistral** via **Ollama**).

---

### Features

* Chunk-based document retrieval using FAISS
* Embedding via `sentence-transformers`
* Context injection to Mistral (7B model)
* Multi-turn conversation support (chat memory)
* Quantitative evaluation with cosine similarity
* Easy to run locally (no cloud dependency)

---

### File Descriptions

| File                                 | Description                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `preprocessing.py`                   | Extracts, cleans, corrects, and chunks raw documents (PDF/TXT) into manageable text pieces for embedding. |
| `embedder.py`                        | Converts all chunks into vector embeddings and saves them into a FAISS index for similarity search.       |
| `retriever.py`                       | Retrieves top-k most relevant chunks from the FAISS index based on user input query.                      |
| `rag_qa.py`                          | Executes a single-turn RAG inference: retrieve → inject into prompt → query Mistral via Ollama.           |
| `rag_qa_memory.py`                   | Enhanced version of `rag_qa.py` that supports follow-up questions and conversation memory.                |
| `rag_eval.py`                        | Runs quantitative evaluation by comparing LLM responses to ground-truth answers using cosine similarity.  |
| `rag_cybersecurity_dataset_en.jsonl` | Dataset of sample QA pairs used for RAG evaluation.                                                       |
| `rag_evaluation_result.xlsx`         | Result of evaluation, showing similarity score between generated and ground-truth answers.                |

---

### Dependencies

Install via pip:

```bash
pip install sentence-transformers faiss-cpu requests
pip install datasets # only for external evaluation dataset (optional)
```

---

### Run the System

1. Preprocess documents:

```bash
python preprocessing.py
```

2. Generate embeddings & index:

```bash
python embedder.py
```

3. Start RAG chatbot:

```bash
python rag_qa.py      # single-turn
python rag_qa_memory.py  # multi-turn with memory
```

4. Evaluate performance:

```bash
python rag_eval.py
```

---

### 💡 Notes

* Requires **Ollama** and **Mistral 7B model** installed locally.
* You can run `ollama run mistral` before starting inference.
* All modules are modular and can be reused or extended.


