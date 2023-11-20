from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from transformers import pipeline
import nltk

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

def extract_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    tagged_words = pos_tag(filtered_words)
    keywords = [word for word, pos in tagged_words if pos.startswith('NN') or pos.startswith('NNP')]
    keyword_freq = Counter(keywords)
    return keyword_freq.most_common(10) 

def extract_keywords_bert(text):
    ner_pipeline = pipeline("ner", model="bert-base-uncased", tokenizer="bert-base-uncased")
    entities = ner_pipeline(text)
    keywords = [entity['word'] for entity in entities if entity['entity'] == 'MISC']
    keyword_freq = Counter(keywords)
    return keyword_freq.most_common(10) 

def generate_graph(keywords):
    G = nx.Graph()
    G.add_nodes_from(keywords)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    keywords_nltk = extract_keywords(text)
    keywords_bert = extract_keywords_bert(text)
    graph_url = generate_graph([word for word, _ in keywords_nltk])
    return render_template('result.html', keywords_nltk=keywords_nltk, keywords_bert=keywords_bert, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
