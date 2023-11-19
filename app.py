from flask import Flask, render_template, request
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords

app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    generate_mind_map(text)
    return render_template('index.html')

def generate_mind_map(text):
    sentences = sent_tokenize(text)
    G = nx.Graph()

    # Adding nodes for each sentence
    G.add_nodes_from(sentences)

    # Plotting the graph
    plt.figure(figsize=(12, 10))  # Adjust figure size here

    # Set node positions using spring layout
    pos = nx.spring_layout(G, seed=42, scale=10)  # Adjust layout parameters here

    # Draw nodes with rounded rectangle shape and contain text within nodes
    node_labels = {sentence: sentence for sentence in sentences}
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=3000, 
        node_shape='s', 
        node_color='lightblue', 
        alpha=0.7
    )

    # Draw node labels as text boxes
    for node, (x, y) in pos.items():
        plt.text(
            x, 
            y, 
            node, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=12
        )

    # Draw edges
    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences):
            if i != j:  # Avoid connecting a sentence to itself
                similarity_score = compute_similarity(sentence1, sentence2)
                if similarity_score > 0.5:  # You can adjust the similarity threshold as needed
                    nx.draw_networkx_edges(G, pos, edgelist=[(sentence1, sentence2)], edge_color='gray', width=1, alpha=0.7)

    plt.title("Generated Mind Map", fontsize=16)
    plt.axis('off')  # Hide axis
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig('static/mind_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_similarity(sentence1, sentence2):
    # Preprocess sentences
    processed_sentence1 = preprocess(sentence1)
    processed_sentence2 = preprocess(sentence2)
    
    # Create spaCy Doc objects
    doc1 = nlp(processed_sentence1)
    doc2 = nlp(processed_sentence2)
    
    # Calculate similarity using spaCy's similarity function
    return doc1.similarity(doc2)


def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize text into words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a sentence
    processed_text = ' '.join(words)
    
    return processed_text

if __name__ == '__main__':
    app.run(debug=True)
