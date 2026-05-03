import pandas as pd
import networkx as nx
from pyvis.network import Network

edges_df = pd.read_csv('data/processed/edges.csv')
nodes_df = pd.read_csv('data/processed/nodes.csv')

net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)

# Словник кольорів для тональностей
color_map = {
    'Захист': '#2ecc71',
    'Звинувачення': '#e74c3c',
    'Нейтрально': '#95a5a6'
}

# граф NetworkX
G = nx.DiGraph()

# вузли
for _, row in nodes_df.iterrows():
    node_id = str(row['id'])
    label = str(row['label'])
    G.add_node(node_id, label=label, title=label)

# ребра зі зв'язками
for _, row in edges_df.iterrows():
    src = str(row['source'])
    tgt = str(row['target'])
    sent = str(row['sentiment'])
    quote = str(row['evidence_quote'])
    
    if not G.has_node(src):
        G.add_node(src, label=src, title=src)
    if not G.has_node(tgt):
        G.add_node(tgt, label=tgt, title=tgt)
        
    edge_color = color_map.get(sent, '#ffffff') # Білий, якщо тональність не розпізнана
    
    # Title буде показуватись при наведенні мишки на ребро (tooltip)
    hover_text = f"[{sent}]\nЦитата: {quote}"
    
    G.add_edge(src, tgt, color=edge_color, title=hover_text)

# Чим більше у людини зв'язків, тим більша її node
centrality = nx.degree_centrality(G)
for node in G.nodes():
    G.nodes[node]['size'] = 15 + (centrality[node] * 100)

# дані з NetworkX у PyVis
net.from_nx(G)
net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

output_file = "kgb_archive_graph.html"
net.write_html(output_file)

print(f"Відкрийте файл {output_file} у вашому браузері.")