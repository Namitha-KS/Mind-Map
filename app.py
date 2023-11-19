# from flask import Flask, render_template, request
# import openai
# import sqlite3

# app = Flask(__name__)

# openai.api_key = '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'

# conn = sqlite3.connect('story_generator.db')
# c = conn.cursor()

# c.execute('''
#    CREATE TABLE IF NOT EXISTS users (
#        id INTEGER PRIMARY KEY,
#        topic TEXT,
#        genre TEXT,
#        characters INTEGER,
#        words INTEGER,
#        additional_info TEXT
#    )
# ''')

# conn.commit()

# def generate_story(user_data):
#    prompt = f"Once upon a time in the {user_data['genre']} genre, there was a story about {user_data['topic']} with {user_data['characters']} characters and a maximum of {user_data['words']} words. {user_data['additional_info']}"

#    response = openai.Completion.create(
#        engine="text-davinci-002",
#        prompt=prompt,
#        max_tokens=100
#    )

#    return response['choices'][0]['text'].strip()

# def generate_image(prompt):
#    res = openai.Image.create(
#        prompt=prompt,
#        n=1,
#        size="256x256",
#    )
#    return res["data"][0]["url"]

# @app.route('/', methods=['GET', 'POST'])
# def home():
#    if request.method == 'POST':
#        topic = request.form['topic']
#        genre = request.form['genre']
#        characters = request.form['characters']
#        words = request.form['words']
#        additional_info = request.form['additional_info']

#        c.execute('''
#            INSERT INTO users (topic, genre, characters, words, additional_info)
#            VALUES (?, ?, ?, ?, ?)
#        ''', (topic, genre, characters, words, additional_info))

#        conn.commit()

#    return render_template('home.html')

# @app.route('/generate', methods=['GET', 'POST'])
# def generate():
#    if request.method == 'POST':
#        user_id = request.form['user_id']

#        c.execute('''
#            SELECT * FROM users WHERE id = ?
#        ''', (user_id,))

#        user_data = c.fetchone()

#        if user_data:
#            story = generate_story({
#                'topic': user_data[1],
#                'genre': user_data[2],
#                'characters': user_data[3],
#                'words': user_data[4],
#                'additional_info': user_data[5]
#            })

#            prompt_for_image = f"Visual representation of the story: {story}"
#            image_url = generate_image(prompt_for_image)

#            return render_template('generate.html', story=story, user_id=user_id, image_url=image_url)

#    return render_template('generate.html')

# def proceed():
#    if request.method == 'POST':
#        user_id = request.form['user_id']
#        user_choice = request.form['user_choice'] # Assuming user_choice contains the selected option

#        next_portion = generate_next_portion(user_choice)
#        story += next_portion

#        prompt_for_image = f"Visual representation of the story: {story}"
#        image_url = generate_image(prompt_for_image)

#        return render_template('generate.html', story=story, user_id=user_id, image_url=image_url)

#    return render_template('generate.html')

# def generate_next_portion(user_choice):
#    prompt = f"Based on the user's choice: {user_choice}, continue the story."

#    response = openai.Completion.create(
#        engine="text-davinci-002",
#        prompt=prompt,
#        max_tokens=100
#    )

#    return response['choices'][0]['text'].strip()


# if __name__ == '__main__':
#    app.run(debug=True)


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

# def generate_mind_map(keywords):
#     G = nx.Graph()
#     G.add_nodes_from(keywords)
#     for i, keyword in enumerate(keywords[:-1]):
#         G.add_edge(keyword, keywords[i+1])
    
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_weight='bold', font_size=10)
#     plt.title("Generated Mind Map")
#     plt.savefig('static/mind_map.png')  # Save the generated mind map
#     plt.close()


# def generate_mind_map(keywords):
#     # Create an empty graph
#     G = nx.Graph()
    
#     # Add nodes from the provided keywords
#     G.add_nodes_from(keywords)
    
#     # Add edges between nodes (forming relationships)
#     for i, keyword in enumerate(keywords[:-1]):
#         G.add_edge(keyword, keywords[i+1])
    
#     # Create a figure for plotting the mind map
#     plt.figure(figsize=(10, 8))
    
#     # Define the layout for the nodes in the graph (customize layout for better readability)
#     pos = nx.spring_layout(G, seed=42)  # Use a seed for consistent layout
    
#     # Draw the graph with improved styling for readability
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_weight='bold', font_size=12,
#             edge_color='gray', width=1, alpha=0.7, arrowsize=15)
    
#     # Set a title for the mind map
#     plt.title("Generated Mind Map", fontsize=16)
    
#     # Save the generated mind map as an image (adjust filename/path as needed)
#     plt.savefig('static/mind_map.png', dpi=300, bbox_inches='tight')  # Save the generated mind map with higher resolution
    
#     # Close the plot to prevent displaying it (if needed)
#     plt.close()

# # Example usage:
# keywords = ["Topic A", "Subtopic 1", "Subtopic 2", "Topic B", "Subtopic 3"]
# generate_mind_map(keywords)

def generate_mind_map(text):
    # Check if the text has less than 1000 words
    words = text.split()
    if len(words) > 1000:
        return "Input text is too long. Please provide text with less than 1000 words."

    # Use TF-IDF to extract main topics
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform([text])
    
    # Apply dimensionality reduction for topic extraction
    svd = TruncatedSVD(n_components=4)
    X_svd = svd.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()
    components = svd.components_
    
    # Extract main topics
    main_topics = []
    for i, topic in enumerate(components):
        topic_keywords = [feature_names[idx] for idx in topic.argsort()[-5:]][::-1]
        main_topics.append(topic_keywords)

    # Generate the mind map
    G = nx.Graph()

    # Add main topics as center blocks
    # for i, topic in enumerate(main_topics):
    #     G.add_node(f"Main Topic {i + 1}", label=", ".join(topic), color='red', shape='rectangle', fontsize=14)

    # # Add subtopics related to each main topic
    # for i, topic in enumerate(main_topics):
    #     for j in range(3):
    #         G.add_node(f"Subtopic {i + 1}-{j + 1}", label=f"{topic[j]}: Short description of {topic[j]}", color='blue', shape='ellipse', fontsize=12)
    #         G.add_edge(f"Main Topic {i + 1}", f"Subtopic {i + 1}-{j + 1}")
    for i, topic in enumerate(main_topics):
        G.add_node(f"Main Topic {i + 1}", label=", ".join(topic), color='red', node_shape=0)  # Use 0 for a circular node shape

    for i, topic in enumerate(main_topics):
        for j in range(3):
            G.add_node(f"Subtopic {i + 1}-{j + 1}", label=f"{topic[j]}: Short description of {topic[j]}", color='blue', node_shape=1)  # Use 1 for a square node shape



    # Create a figure for plotting the mind map
    plt.figure(figsize=(12, 10))
    
    # Define positions for main topics and subtopics
    positions = {
        f"Main Topic {i + 1}": (0, i * 2) for i in range(4)
    }
    for i in range(4):
        for j in range(3):
            positions[f"Subtopic {i + 1}-{j + 1}"] = ((j + 1) * 2, i * 2 - 0.5 + (j - 1) * 0.5)

    # Draw the graph
    nx.draw(G, positions, with_labels=True, node_size=800, font_weight='bold', font_size=10, node_color=[G.nodes[n]['color'] for n in G.nodes()], node_shape=[G.nodes[n]['shape'] for n in G.nodes()])
    
    # Set a title for the mind map
    plt.title("Generated Mind Map", fontsize=16)
    
    # Save the generated mind map as an image
    plt.savefig('static/mind_map.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
sample_text = "Your text here. Ensure it has less than 1000 words."
generate_mind_map(sample_text)


if __name__ == '__main__':
    app.run(debug=True)