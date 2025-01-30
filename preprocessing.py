# Load the corpus data
with open('corpus.txt', 'r', encoding='utf-8') as file:
    corpus_data = file.read()

# Print the first 500 characters to confirm the data is loaded correctly
print(corpus_data[:500])  # Adjust the number for more or less preview
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

# Download the necessary resources
nltk.download('punkt_tab')



def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Return the cleaned and filtered text as a list of words
    return ' '.join(filtered_tokens)

# Preprocess the entire corpus
cleaned_data = preprocess_text(corpus_data)

# Print the first 500 characters of the cleaned data
print(cleaned_data[:500])

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000) 

# Convert the result to a dense array (optional, depending on your use case)


# Split the corpus into a list of documents (each line treated as a separate document)
cleaned_data_list = cleaned_data.split('\n')

# Apply TF-IDF vectorizer
X = vectorizer.fit_transform(cleaned_data_list)



# Print the shape of the resulting TF-IDF matrix
print(f"Shape of TF-IDF matrix: {X.shape}")
