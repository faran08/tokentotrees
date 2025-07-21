import plotly.graph_objs as go

from models.language_model import get_top_n_next_words


def initialize_graph(initial_sentence, node_sequences, tree, edge_probs, positions):
    node_sequences.clear()
    tree.clear()
    edge_probs.clear()
    positions.clear()
    node_sequences[0] = initial_sentence  # Store the full sentence
    tree[0] = []
    positions[0] = (0, 0)


# def generate_graph(clicked_node, node_sequences, tree, edge_probs, positions, model, tokenizer):
#     if clicked_node is not None:
#         sequence = node_sequences[clicked_node]
#         # 🔍 Log the sequence we're working with
#         print(f"Clicked Node: {clicked_node} — Sequence: '{sequence}'")
#         next_words = get_top_n_next_words(sequence, model, tokenizer, n=3)

#         for word, prob in next_words:
#             new_sequence = sequence + " " + word
#             new_node_id = len(node_sequences)
#             node_sequences[new_node_id] = new_sequence
#             tree[clicked_node].append(new_node_id)
#             tree[new_node_id] = []
#             edge_probs[(clicked_node, new_node_id)] = prob

#     # Compute positions (tree layout)
#     positions.clear()
#     queue = [(0, 0, 0)]  # node, depth, y_offset
#     y_offsets = {0: 0}
#     while queue:
#         node, depth, _ = queue.pop(0)
#         if depth not in y_offsets:
#             y_offsets[depth] = 0
#         positions[node] = (depth, y_offsets[depth] -
#                            len(tree.get(node, [])) / 2)
#         y_offsets[depth] += 1
#         for child in tree.get(node, []):
#             queue.append((child, depth + 1, positions[node][1]))

#     # Prepare Plotly traces
#     x_vals = [pos[0] for pos in positions.values()]
#     y_vals = [pos[1] for pos in positions.values()]
#     texts = [seq.split()[-1]
#              for seq in node_sequences.values()]  # Still show last word only

#     node_trace = go.Scatter(
#         x=x_vals, y=y_vals,
#         mode='markers+text',
#         text=texts,
#         textposition='top center',
#         marker=dict(
#             size=20,
#             color='#1f77b4',
#             line=dict(width=2, color='DarkSlateGrey')
#         ),
#         hoverinfo='text',
#         # Full sequence on hover
#         hovertext=[f"{seq} (ID: {i})" for i, seq in node_sequences.items()]
#     )

#     edge_traces = []
#     for (parent, child), prob in edge_probs.items():
#         parent_pos = positions[parent]
#         child_pos = positions[child]
#         edge_traces.append(go.Scatter(
#             x=[parent_pos[0], child_pos[0], None],
#             y=[parent_pos[1], child_pos[1], None],
#             mode='lines',
#             line=dict(width=prob * 10, color='#888'),
#             hoverinfo='text',
#             hovertext=f'Prob: {prob:.3f}'
#         ))

#     fig = go.Figure(data=[node_trace] + edge_traces)
#     fig.update_layout(
#         showlegend=False,
#         clickmode='event+select',
#         xaxis=dict(showgrid=False, zeroline=False),
#         yaxis=dict(showgrid=False, zeroline=False),
#         margin=dict(l=20, r=20, t=20, b=20),
#         plot_bgcolor='rgba(245, 245, 245, 1)',
#         transition=dict(duration=500, easing='cubic-in-out')
#     )

#     return fig


def generate_graph(clicked_node, node_sequences, tree, edge_probs, positions, model, tokenizer_for_ids):
    if clicked_node is not None:
        sequence = node_sequences[clicked_node]
        print(f"Clicked Node: {clicked_node} — Sequence: '{sequence}'")
        next_words = get_top_n_next_words(sequence, model, tokenizer_for_ids, n=3)

        for word, prob in next_words:
            new_sequence = sequence + " " + word
            new_node_id = len(node_sequences)
            node_sequences[new_node_id] = new_sequence
            tree[clicked_node].append(new_node_id)
            tree[new_node_id] = []
            edge_probs[(clicked_node, new_node_id)] = prob

    # Compute positions (tree layout)
    positions.clear()
    queue = [(0, 0, 0)]  # node, depth, y_offset
    y_offsets = {0: 0}
    while queue:
        node, depth, _ = queue.pop(0)
        if depth not in y_offsets:
            y_offsets[depth] = 0
        positions[node] = (depth, y_offsets[depth] - len(tree.get(node, [])) / 2)
        y_offsets[depth] += 1
        for child in tree.get(node, []):
            queue.append((child, depth + 1, positions[node][1]))

    # Prepare Plotly traces
    x_vals = [pos[0] for pos in positions.values()]
    y_vals = [pos[1] for pos in positions.values()]
    texts = [seq.split()[-1] for seq in node_sequences.values()]  # Show last word

    # Build hover texts with extra info
    hover_texts = []
    for node_id, seq in node_sequences.items():
        last_word = seq.split()[-1]
        parent = 'None'
        parent_ids = [p for p, children in tree.items() if node_id in children]
        if parent_ids:
            parent = str(parent_ids[0])
        prob = edge_probs.get((int(parent) if parent != 'None' else -1, node_id), 0.0)
        token_id = tokenizer_for_ids.convert_tokens_to_ids(last_word)
        hover_texts.append(
            f"<b>Word:</b> {last_word}<br>"
            f"<b>Full sequence:</b> {seq}<br>"
            f"<b>Node ID:</b> {node_id}<br>"
            f"<b>Parent ID:</b> {parent}<br>"
            f"<b>Token ID:</b> {token_id}<br>"
            f"<b>Prob:</b> {prob:.4f}"
        )

    node_trace = go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        text=texts,
        textposition='top center',
        marker=dict(
            size=20,
            color='#1f77b4',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        hovertext=hover_texts,
    )

    edge_traces = []
    for (parent, child), prob in edge_probs.items():
        parent_pos = positions[parent]
        child_pos = positions[child]
        edge_traces.append(go.Scatter(
            x=[parent_pos[0], child_pos[0], None],
            y=[parent_pos[1], child_pos[1], None],
            mode='lines',
            line=dict(width=prob * 10, color='#888'),
            hoverinfo='text',
            hovertext=f'Prob: {prob:.3f}'
        ))

    fig = go.Figure(data=[node_trace] + edge_traces)
    fig.update_layout(
        showlegend=False,
        clickmode='event+select',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(245, 245, 245, 1)',
        transition=dict(duration=500, easing='cubic-in-out')
    )

    return fig
