import ollama

def ask_mistral(question, context):
    prompt = f"""You are a legal assistant. Answer the following question using the context below:
    
Context:
{context}

Question:
{question}

Answer:"""

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()
