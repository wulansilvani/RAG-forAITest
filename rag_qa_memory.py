import requests
from retriever import retrieve_chunks

# Save chat history
chat_history = []

def build_prompt_with_memory(history, current_question, context):
    conversation = ""
    for turn in history:
        conversation += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    conversation += f"User: {current_question}"

    prompt = f"""You are a cybersecurity assistant. Use the following context and conversation history to answer the latest question.\n\nContext:\n{context}\n\nConversation:\n{conversation}\n\nAnswer:"""
    return prompt

def ask_mistral_with_memory(current_question):
    context_chunks = retrieve_chunks(current_question, top_k=5)
    context = "\n\n".join(context_chunks)
    prompt = build_prompt_with_memory(chat_history, current_question, context)

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    answer = response.json()["response"].strip()
    chat_history.append({"user": current_question, "assistant": answer})
    return answer

# === Loop untuk tanya jawab ===
if __name__ == "__main__":
    print("CyberSec Chat with Mistral (type 'exit' to quit)\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        reply = ask_mistral_with_memory(question)
        print(f"\n Mistral: {reply}\n")
