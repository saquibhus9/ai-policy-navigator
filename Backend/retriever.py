import os
import faiss
import pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Directories
pdf_dir = "pdfs"
vector_dir = "vectordb"
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)

# Load Sentence Transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384  # match model

# Paths
index_path = os.path.join(vector_dir, "index.faiss")
data_path = os.path.join(vector_dir, "data.pkl")

# Global cache
index = faiss.IndexFlatL2(embedding_dim)
text_chunks = []

if os.path.exists(index_path) and os.path.exists(data_path):
    index = faiss.read_index(index_path)
    with open(data_path, "rb") as f:
        text_chunks = pickle.load(f)

# ✅ In-memory uploaded PDF text
uploaded_pdf_text = ""


def save_pdf_and_index(filename, content):
    global uploaded_pdf_text

    path = os.path.join(pdf_dir, filename)
    with open(path, "wb") as f:
        f.write(content)

    doc = fitz.open(stream=content, filetype="pdf")
    all_text = ""
    for page in doc:
        txt = page.get_text()
        if txt:
            all_text += txt + "\n"
    doc.close()

    uploaded_pdf_text = all_text  # Save in memory

    # Chunk and embed
    chunks = [all_text[i:i + 500].strip() for i in range(0, len(all_text), 500) if all_text[i:i + 500].strip()]
    if not chunks:
        print(f"[Warning] No text extracted from {filename}")
        return

    embeddings = embedder.encode(chunks)
    index.add(embeddings)
    text_chunks.extend(chunks)

    # Save index and data
    faiss.write_index(index, index_path)
    with open(data_path, "wb") as f:
        pickle.dump(text_chunks, f)

    print(f"[Info] Indexed {len(chunks)} chunks from {filename}")


def get_similar_chunks(query, k=3):
    global uploaded_pdf_text

    if not uploaded_pdf_text.strip():
        return ""  # No uploaded PDF

    # Chunk uploaded PDF dynamically
    chunks = [uploaded_pdf_text[i:i + 500].strip() for i in range(0, len(uploaded_pdf_text), 500) if uploaded_pdf_text[i:i + 500].strip()]
    if not chunks:
        return ""

    embeddings = embedder.encode(chunks)
    query_vec = embedder.encode([query])
    index_local = faiss.IndexFlatL2(embedding_dim)
    index_local.add(embeddings)

    distances, indices = index_local.search(query_vec, k)
    valid_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

    return "\n\n".join(chunks[i] for i in valid_idxs) if valid_idxs else ""
