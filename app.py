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

    # Compute node sizes based on text length
    node_sizes = [len(sentence) * 100 for sentence in sentences]  # Adjust multiplier to control node size
    max_node_size = max(node_sizes)

    # Draw nodes with text
    for sentence, (x, y), size in zip(sentences, pos.values(), node_sizes):
        plt.text(
            x, 
            y, 
            sentence, 
            bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='black', pad=0.5),
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=12
        )
        node_width = size / max_node_size * 3000  # Adjust the multiplier for node width
        plt.scatter(x, y, s=0, marker='s')  # Invisible marker to define node size
        plt.scatter(x, y, s=node_width, c='lightblue', alpha=0.7, edgecolor='black')

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
