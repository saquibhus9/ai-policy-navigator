# import os
# import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle

# DATA_FOLDER = "data/documents"
# INDEX_PATH = "embeddings/faiss_index.bin"
# DOCS_PATH = "embeddings/docs.pkl"

# def extract_text_from_pdfs(folder=DATA_FOLDER):
#     texts = []
#     files = []
#     for file in os.listdir(folder):
#         if file.endswith(".pdf"):
#             doc = fitz.open(os.path.join(folder, file))
#             text = ""
#             for page in doc:
#                 text += page.get_text()
#             texts.append(text)
#             files.append(file)
#     return texts, files

# def create_vector_store():
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     texts, files = extract_text_from_pdfs()
#     embeddings = model.encode(texts)
#     index = faiss.IndexFlatL2(embeddings[0].shape[0])
#     index.add(embeddings)
    
#     with open(DOCS_PATH, "wb") as f:
#         pickle.dump((texts, files), f)
#     faiss.write_index(index, INDEX_PATH)
#     print("✅ Vector store created.")

# if __name__ == "__main__":
#     create_vector_store()
import os
import pickle
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "Data/documents"
DOCS_PATH = "embeddings/docs.pkl"
INDEX_PATH = "embeddings/faiss.index"


def extract_text_from_pdfs(folder=DATA_FOLDER):
    texts = []
    files = []
    print(f"📂 Scanning folder: {folder}")
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            print(f"➡️ Processing {file}")
            doc = fitz.open(os.path.join(folder, file))
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
            files.append(file)
    return texts, files


def create_vector_store():
    print("🔍 Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📄 Extracting text from PDFs...")
    texts, files = extract_text_from_pdfs()

    if not texts:
        print("❌ No text extracted from PDFs. Please check if the PDF folder is correct or contains readable files.")
        return

    print(f"✅ Extracted {len(texts)} documents.")

    print("🧠 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("📦 Initializing FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("💾 Saving FAISS index and documents...")
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump((texts, files), f)

    faiss.write_index(index, INDEX_PATH)

    print("✅ Vector store created.")


def load_vector_store():
    with open(DOCS_PATH, "rb") as f:
        texts, files = pickle.load(f)

    index = faiss.read_index(INDEX_PATH)
    return texts, files, index
