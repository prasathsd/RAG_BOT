import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from spellchecker import SpellChecker
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load spaCy transformer model for better NER
try:
    nlp = spacy.load("en_core_web_trf")  
except OSError:
    print("[Error] spaCy model 'en_core_web_trf' not found. Run: python -m spacy download en_core_web_trf")
    exit()

# Load spell checker
spell = SpellChecker()

def correct_query(query):
    words = query.split()
    corrected_words = []
    
    for word in words:
        lowercase_word = word.lower()
        suggested_correction = spell.correction(lowercase_word)
        spell_candidates = spell.candidates(lowercase_word)

        print(f"Word: {word} | Candidates: {spell_candidates}")  # Debugging output

        if word.istitle():
            corrected_word = suggested_correction.capitalize() if suggested_correction else word
        else:
            corrected_word = suggested_correction if suggested_correction else word

        # Fallback correction using TextBlob if SpellChecker fails
        if corrected_word.lower() == word.lower():
            corrected_word = str(TextBlob(word.lower()).correct())

        corrected_words.append(corrected_word)
        print(f"Original: {word} -> Suggested: {corrected_word}")  

    return " ".join(corrected_words)

def extract_entities(query):
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

def retrieve_top_documents(query, corpus_data, vectorizer, top_n=3, similarity_threshold=0.15):
    corpus_vectorized = vectorizer.transform(corpus_data)
    query_vectorized = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vectorized, corpus_vectorized)[0]

    # Get top documents based on similarity
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    retrieved_docs = []
    for i in top_indices:
        score = similarity_scores[i]
        if score >= similarity_threshold:  # Only consider relevant docs
            retrieved_docs.append({
                "index": int(i),
                "document": corpus_data[i],
                "similarity_score": float(score)
            })

    # Debug: Print retrieved documents before passing to model
    print("\n🔍 **Retrieved Documents for Query:**", query)
    for doc in retrieved_docs:
        print(f"📄 [Doc Index: {doc['index']}] (Similarity: {doc['similarity_score']:.4f})\n{doc['document']}\n")

    return retrieved_docs if retrieved_docs else ["No relevant documents found."]





def load_t5_model():
    model_name = "google/flan-t5-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ Model and Tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ ERROR: Failed to load model/tokenizer: {e}")
        return None, None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_t5_model()

if model is None or tokenizer is None:
    print("❌ ERROR: Model or Tokenizer failed to load. Exiting...")
    exit()

model.to(device)

def generate_response(query, top_documents, model, tokenizer, context=""):
    if not top_documents:
        return "No valid documents found."

    # Extract only document text
    valid_docs = [doc['document'] if isinstance(doc, dict) else doc for doc in top_documents]

    # Ensure at least some valid context
    if not valid_docs:
        return "No relevant documents found."

    # Prepare full input for model
    full_context = f"Query: {query}\nContext:\n" + "\n\n".join(valid_docs)

    # Tokenization
    inputs = tokenizer(full_context, return_tensors="pt", max_length=2048, truncation=True)

    # Generate response
    outputs = model.generate(inputs['input_ids'], max_length=250, num_return_sequences=1)

    # Decode and return
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text if response_text.strip() else "No meaningful response generated."

# Load corpus data
corpus_path = 'corpus.txt'
if not os.path.exists(corpus_path):
    print("[Error] Corpus file 'corpus.txt' not found.")
    exit()

with open(corpus_path, 'r', encoding='utf-8') as file:
    corpus_data = file.read().split('\n')

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(corpus_data)

# Example query
query = "Whar is the capital of Indian?"
corrected_query = correct_query(query)
entities = extract_entities(corrected_query)
top_documents = retrieve_top_documents(corrected_query, corpus_data, vectorizer)
response = generate_response(corrected_query, top_documents, model, tokenizer)

print(f"Original Query: {query}")
print(f"Corrected Query: {corrected_query}")
print(f"Extracted Entities: {entities}")
print("\nBot Response:\n", response)
