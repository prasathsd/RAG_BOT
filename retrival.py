import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
try:
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load FAISS index
try:
    print("Loading FAISS index...")
    index = faiss.read_index("vector_index.faiss")
    print("Index loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit()

# Load corpus
try:
    print("Loading corpus...")
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    print("Corpus loaded successfully!")
except Exception as e:
    print(f"Error loading corpus: {e}")
    exit()

def retrieve_top_k(query, k=3):
    print(f"Query: {query}")
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    results = [corpus[i] for i in indices[0]]
    
    return results

print(retrieve_top_k("neural networks"))