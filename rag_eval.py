import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Load dataset and model
dataset_path = Path("rag_cybersecurity_dataset_en.jsonl")
records = [json.loads(line) for line in open(dataset_path, encoding="utf-8")]
df = pd.DataFrame(records)
model = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_mistral(question, context):
    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    response = requests.post(OLLAMA_URL, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip()

# Eval
results = []
for row in df.itertuples():
    pred = ask_mistral(row.question, row.context)
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_truth = model.encode(row.answer, convert_to_tensor=True)
    score = util.cos_sim(emb_pred, emb_truth).item()
    results.append({
        "question": row.question,
        "ground_truth": row.answer,
        "generated_answer": pred,
        "similarity_score": round(score, 4)
    })

# Output
pd.DataFrame(results).to_csv("rag_evaluation_result.csv", index=False)
print("Evaluation complete! Results saved to rag_evaluation_result.csv")
