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
    
    for i, word in enumerate(words):
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

def retrieve_top_documents(query, corpus_data, vectorizer, top_n=3):
    corpus_vectorized = vectorizer.transform(corpus_data)
    query_vectorized = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vectorized, corpus_vectorized)
    top_doc_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    return [{"index": i, "document": corpus_data[i], "similarity_score": similarity_scores[0][i]} for i in top_doc_indices]

def load_t5_model():
    model_name = "google/flan-t5-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("✅ Model and Tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ ERROR: Failed to load model/tokenizer: {e}")
        return None, None  # Return None if model/tokenizer loading fails

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_t5_model()

# Ensure model and tokenizer are properly loaded
if model is None or tokenizer is None:
    print("❌ ERROR: Model or Tokenizer failed to load. Exiting...")
    exit()

model.to(device)

def generate_response(query, top_documents, model, tokenizer, context=""):
    if tokenizer is None or model is None:
        print("❌ ERROR: Tokenizer or Model is None in generate_response()")
        return "Error: Model not initialized properly."
    
    if isinstance(top_documents, list) and all(isinstance(doc, str) for doc in top_documents):
        top_documents = [{"document": doc} for doc in top_documents]

    full_context = f"{context}\n{query}\n" + "\n".join([doc['document'] for doc in top_documents])
    
    inputs = tokenizer(full_context, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
query = "Whar is the capital of India?"
corrected_query = correct_query(query)
entities = extract_entities(corrected_query)
top_documents = retrieve_top_documents(corrected_query, corpus_data, vectorizer)
response = generate_response(corrected_query, top_documents, model, tokenizer)


print(f"Original Query: {query}")
print(f"Corrected Query: {corrected_query}")
print(f"Extracted Entities: {entities}")
print("\nBot Response:\n", response)
