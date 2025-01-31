import wikipediaapi

def fetch_wikipedia_pages(topic_list, num_sentences=500):
    # Initialize Wikipedia with correct user-agent syntax
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='Chatbot1(prasath1503dpi@gmail.com)')

    
    corpus = {}

    for topic in topic_list:
        page = wiki_wiki.page(topic)
        if page.exists():
            text = page.text
            sentences = text.split('. ')
            corpus[topic] = '. '.join(sentences[:num_sentences])  # Limit text size
        else:
            print(f"Page '{topic}' not found.")

    return corpus

if __name__ == "__main__":
    topics = ["Machine Learning", "Artificial Intelligence", "Neural Networks","Quant Finance"]  # Example topics
    corpus_data = fetch_wikipedia_pages(topics)
    
    with open("corpus.txt", "w", encoding="utf-8") as f:
        for topic, content in corpus_data.items():
            f.write(f"### {topic} ###\n{content}\n\n")

    print("✅ Wikipedia data collected and saved to corpus.txt")
