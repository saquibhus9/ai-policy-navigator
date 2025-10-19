# ask_from_upload.py

from retriever import save_pdf_and_index, get_similar_chunks
from ask_mistral import ask_mistral

# Simulate PDF upload
with open("Backend/pdfs/file1.pdf", "rb") as f:
    content = f.read()
    save_pdf_and_index("file1.pdf", content)

# Now ask a question
query = "What is this policy about?"
context = get_similar_chunks(query)

if not context:
    print("⚠️ No relevant content found in uploaded PDF.")
else:
    answer = ask_mistral(context, query)
    print(f"\n📌 Q: {query}\n\n💬 A: {answer}")
