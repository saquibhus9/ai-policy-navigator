 # test_vector_store.py
import os
import pickle
import faiss

def load_vector_store():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Base directory: {base_dir}")

    docs_path = os.path.join(base_dir, "embeddings", "docs.pkl")
    index_path = os.path.join(base_dir, "embeddings", "faiss_index.bin")

    
    print(f"🔍 Loading docs from: {docs_path}")
    print(f"🔍 Loading index from: {index_path}")

    with open(docs_path, "rb") as f:
        texts, files = pickle.load(f)

    index = faiss.read_index(index_path)

    return texts, files, index

if __name__ == "__main__":
    texts, files, index = load_vector_store()
    print(f"✅ Loaded {len(texts)} documents with {index.ntotal} vectors.")
