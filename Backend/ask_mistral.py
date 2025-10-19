import subprocess

def ask_mistral(context: str, query: str) -> str:
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Answer in simple language:"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding='utf-8'  # <- ADD THIS
    )
    return result.stdout.strip()
