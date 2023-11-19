from flask import Flask, render_template, request
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    keywords = extract_keywords(text)
    generate_mind_map(keywords)
    return render_template('index.html')

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.pos_ in ["NOUN", "VERB", "ADJ"]]
    return keywords

def generate_mind_map(keywords):
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes from the provided keywords
    G.add_nodes_from(keywords)
    
    # Add edges between nodes (forming relationships)
    for i, keyword in enumerate(keywords[:-1]):
        G.add_edge(keyword, keywords[i+1])
    
    # Create a figure for plotting the mind map
    plt.figure(figsize=(10, 8))
    
    # Define the layout for the nodes in the graph (customize layout for better readability)
    pos = nx.spring_layout(G, seed=42)  # Use a seed for consistent layout
    
    # Draw the graph with improved styling for readability
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_weight='bold', font_size=12,
            edge_color='gray', width=1, alpha=0.7, arrowsize=15)
    
    # Set a title for the mind map
    plt.title("Generated Mind Map", fontsize=16)
    
    # Save the generated mind map as an image (adjust filename/path as needed)
    plt.savefig('static/mind_map.png', dpi=300, bbox_inches='tight')  # Save the generated mind map with higher resolution
    
    # Close the plot to prevent displaying it (if needed)
    plt.close()



# Example usage:
sample_text = "Your text here. Ensure it has less than 1000 words."
generate_mind_map(sample_text)


if __name__ == '__main__':
    app.run(debug=True)
