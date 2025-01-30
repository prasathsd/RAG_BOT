# RAG Chatbot API

This is a **Retrieval-Augmented Generation (RAG) Chatbot API** built with **Flask**. It retrieves relevant documents based on user queries and generates responses using a pre-trained language model.

## Features
- Accepts user queries via a POST request.
- Retrieves relevant document chunks using a retrieval function.
- Uses a transformer-based model to generate responses.
- Provides responses in JSON format.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- Flask
- Transformers (Hugging Face)
- Scikit-learn
- spaCy
- TextBlob
- SpellChecker

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/YOUR_USERNAME/rag-chatbot-api.git
   cd rag-chatbot-api
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download the spaCy model:
   ```sh
   python -m spacy download en_core_web_trf
   ```


### API Endpoints
#### 1. Home Endpoint
- **URL:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  "Welcome to the RAG Chatbot API!"
  ```

#### 2. Chat Endpoint
- **URL:** `/chat`
- **Method:** `POST`
- **Request Body:**
  ```json
  { "query": "What is AI?" }
  ```
- **Response:**
  ```json
  {
    "query": "What is AI?",
    "response": "Artificial Intelligence (AI) refers to..."
  }
  ```

## Deployment
To deploy the API using **Gunicorn**:
```sh
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Contributing
Feel free to fork and contribute! Create a pull request with improvements.


