# Mind-Map

# MindMap Generator

The MindMap Generator is a web application built using Flask, designed to extract keywords from user-input text and visualize them in a network graph, providing an intuitive representation of the central concepts within the text. Mind maps serve as powerful visual tools that aid in organizing thoughts, concepts, and information in a non-linear and intuitive manner. 

## Features

### Extract Keywords

- The application extracts keywords from the provided text using NLTK and BERT-based NER (Named Entity Recognition) from the Hugging Face Transformers library.

### Network Graph Generation

- Generates a network graph using NetworkX and Matplotlib to visualize the extracted keywords.
- Links the keywords to a root node labeled "MindMap" to represent the central concept.

## Libraries Used

- [Hugging Face Transformers](https://huggingface.co/transformers/): Utilized for BERT-based Named Entity Recognition (NER) to extract keywords.
- [Flask](https://flask.palletsprojects.com/en/2.0.x/): A web framework for Python used to build the application.
- [NLTK (Natural Language Toolkit)](https://www.nltk.org/): Used for tokenization, part-of-speech tagging, and stopword removal in text processing.
- [NetworkX](https://networkx.org/): Library for the creation, manipulation, and study of complex networks.
- [Matplotlib](https://matplotlib.org/): A plotting library used for generating the network graph visualization.

## Generated Mind-Maps

![App Screenshot](map.png)

This is a feature updation of the projet that I am trying to build, which will help students learn anything much effectively than the conventional way of teaching : https://x.com/13_cs2/status/1692441104199594490?s=20

## Usage
1. Clone the Repository
 '''git clone https://github.com/'''

Install Dependencies: Install necessary Python libraries by running pip install -r requirements.txt.

Run the Application: Start the Flask-based web application with python app.py.

Input Text: Enter text into the provided form and click "Generate MindMap" to trigger keyword extraction and graph visualization.

Graph Visualization: Visualize the extracted keywords and their connections in the generated network graph displayed on the web interface.
