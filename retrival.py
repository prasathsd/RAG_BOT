import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index("vector_index.faiss")
print("Index loaded successfully!")

# Load corpus
print("Loading corpus...")
with open("corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
print(f"Corpus loaded successfully! ({len(corpus)} documents)")

def retrieve_top_k(query, k=3):
    print(f"\n🔍 Query: {query}")
    query_embedding = embedding_model.encode([query]).astype(np.float32)

    # Search in FAISS
    distances, indices = index.search(query_embedding, k)
    
    print(f"📌 Retrieved indices: {indices}")
    print(f"📏 Distances: {distances}")

    # Fetch full document text for each retrieved index
    results = [corpus[i] for i in indices[0] if i < len(corpus)]

    # Filter out single-word or too-short responses
    filtered_results = [doc for doc in results if len(doc.split()) > 5]

    # If filtering removes all documents, return at least one
    final_response = filtered_results[0] if filtered_results else results[0]

    print(f"✅ Final Response: {final_response}")
    
    return final_response


# Example query test
if __name__ == "__main__":
    print(retrieve_top_k("what is supervised learning?"))
