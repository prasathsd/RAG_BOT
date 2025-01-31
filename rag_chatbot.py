from flask import Flask, request, jsonify
from retrival import retrieve_top_k  # Importing retrieval function

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    
    if not data or "query" not in data:
        return jsonify({"error": "Query not provided"}), 400

    query = data["query"]
    retrieved_text = retrieve_top_k(query)  # Get the retrieved response

    print(f"🚀 Returning Response: {retrieved_text}")  # Debugging log

    return jsonify({"query": query, "response": retrieved_text})

if __name__ == "__main__":
    app.run(debug=True)
