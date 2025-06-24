import requests
from retriever import retrieve_chunks

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
TOP_K = 5

def ask_mistral(question: str) -> str:
    context_chunks = retrieve_chunks(question, top_k=TOP_K)
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a cybersecurity assistant. Use the following context to answer the question accurately.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    return data.get("response", "[No response received]")

# Example
if __name__ == "__main__":
    user_question = input("Ask something about cybersecurity: ")
    answer = ask_mistral(user_question)
    print("\n Mistral says:\n")
    print(answer)
