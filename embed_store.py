import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load corpus
with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read().split("\n")

# Generate embeddings
corpus_embeddings = embedding_model.encode(corpus)

# Convert to NumPy array
embedding_dim = corpus_embeddings.shape[1]
corpus_embeddings = np.array(corpus_embeddings, dtype=np.float32)

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(corpus_embeddings)

# Save index
faiss.write_index(index, "vector_index.faiss")

# Save corpus mapping
with open("corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)

print("Vector database created and stored successfully!")
