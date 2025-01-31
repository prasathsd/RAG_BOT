import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Load corpus
try:
    print("Loading corpus...")
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    print(f"Corpus loaded successfully! ({len(corpus)} documents)")
except Exception as e:
    print(f"Error loading corpus: {e}")
    exit()

# Compute embeddings
print("Computing embeddings for corpus...")
embeddings = embedding_model.encode(corpus, convert_to_numpy=True)
print("Embeddings computed!")

# Create FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
print("FAISS index built successfully!")

# Save FAISS index
faiss.write_index(index, "vector_index.faiss")
print("FAISS index saved as 'vector_index.faiss'!")
