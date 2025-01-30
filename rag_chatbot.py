from flask import Flask, request, jsonify
from retrival import retrieve_top_k  # Import retrieval function
from gpt_torch import generate_response, load_t5_model  # Import response generator and model loader

app = Flask(__name__)

# Load model and tokenizer globally
model, tokenizer = load_t5_model()

@app.route('/')
def home():
    return "Welcome to the RAG Chatbot API!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant documents
    retrieved_chunks = retrieve_top_k(user_query)

    # Ensure retrieved_chunks is a list of dicts
    if isinstance(retrieved_chunks, list) and all(isinstance(chunk, str) for chunk in retrieved_chunks):
        retrieved_chunks = [{"document": chunk} for chunk in retrieved_chunks]  

    # Generate response using retrieved context
    response = generate_response(user_query, retrieved_chunks, model, tokenizer)

    return jsonify({
        "query": user_query,
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
