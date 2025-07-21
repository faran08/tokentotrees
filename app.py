import time
from flask import Flask, jsonify, render_template, request
import json

import plotly
from models.language_model import load_model, get_top_n_next_words
from utils.graph_utils import initialize_graph, generate_graph

app = Flask(__name__)
app.debug = True
# Load the model and tokenizer globally
model, tokenizer = load_model()

# Global data structures for the graph
node_sequences = {}
tree = {}
edge_probs = {}
positions = {}


@app.route('/', methods=['GET', 'POST'])
def index():
    global node_sequences, tree, edge_probs, positions
    if request.method == 'POST':
        # Changed from 'initial_word'
        initial_sentence = request.form['initial_sentence'].strip()
        if initial_sentence:
            initialize_graph(initial_sentence, node_sequences,
                             tree, edge_probs, positions)
            fig = generate_graph(None, node_sequences, tree,
                                 edge_probs, positions, model, tokenizer)
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('index.html', graph_json=graph_json)
    return render_template('index.html', graph_json=None)


@app.route('/update_graph', methods=['POST'])
def update_graph():
    global node_sequences, tree, edge_probs, positions
    clicked_node = int(request.form['clicked_node'])
    # — Log to your Flask console —
    app.logger.debug(f"node_sequences = {node_sequences!r}")
    # or simply
    print("node_sequences:", node_sequences)
    fig = generate_graph(clicked_node, node_sequences, tree,
                         edge_probs, positions, model, tokenizer)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json


@app.route('/auto_generate', methods=['POST'])
def auto_generate():
    global node_sequences, tree, edge_probs, positions
    # Default to 0 (root) if not provided
    current_node = int(request.form.get('current_node', 0))
    sequence = node_sequences.get(
        current_node, request.form.get('initial_sentence', '').strip())

    # If starting fresh, initialize with the initial sentence
    if current_node == 0 and sequence:
        initialize_graph(sequence, node_sequences, tree, edge_probs, positions)

    # Add one new token
    next_words = get_top_n_next_words(
        sequence, model, tokenizer, n=1)  # Take top 1
    if next_words:
        word, prob = next_words[0]
        new_sequence = sequence + " " + word
        new_node_id = len(node_sequences)
        node_sequences[new_node_id] = new_sequence
        tree[current_node].append(new_node_id)
        tree[new_node_id] = []
        edge_probs[(current_node, new_node_id)] = prob
        current_node = new_node_id  # Update current node
    else:
        return jsonify({'done': True})  # Signal end if no more predictions

    # Generate and return the updated graph
    fig = generate_graph(None, node_sequences, tree,
                         edge_probs, positions, model, tokenizer)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'graph': graph_json, 'current_node': current_node})


if __name__ == '__main__':
    app.run(debug=True)
