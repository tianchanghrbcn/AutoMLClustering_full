import matplotlib.pyplot as plt
import networkx as nx

# Nodes and edges
nodes = [
    "Initial Dataset",
    "Dataset with Errors",
    "Cleaned Dataset",
    "Clustered Data",
    "Tuned Cluster Results",
    "Validated Results",
    "Normalized Metrics",
    "Scheme Results",
    "Dataset Results"
]

edges = [
    ("Initial Dataset", "Dataset with Errors", "Inject Errors"),
    ("Dataset with Errors", "Cleaned Dataset", "Data Cleaning"),
    ("Cleaned Dataset", "Clustered Data", "Clustering"),
    ("Clustered Data", "Tuned Cluster Results", "Automated Parameter Tuning"),
    ("Tuned Cluster Results", "Validated Results", "Analyze Validity"),
    ("Validated Results", "Normalized Metrics", "Compute Metrics"),
    ("Normalized Metrics", "Scheme Results", "Evaluate Scores"),
    ("Scheme Results", "Dataset Results", "Identify Best Scheme")
]

# Create graph
G = nx.DiGraph()
for node in nodes:
    G.add_node(node)
for edge in edges:
    G.add_edge(edge[0], edge[1], step=edge[2])

# Adjust node positions with reduced vertical spacing
pos = {
    "Initial Dataset": (0, 10),
    "Dataset with Errors": (4, 10),  # Adjust x and y for spacing
    "Cleaned Dataset": (8, 10),
    "Clustered Data": (12, 10),
    "Tuned Cluster Results": (16, 10),
    "Validated Results": (16, 8),
    "Normalized Metrics": (12, 8),
    "Scheme Results": (8, 8),
    "Dataset Results": (4, 8),
}

# Adjust figure size for better proportions
plt.figure(figsize=(20, 6))  # Adjust width and height

# Draw graph
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=0  # Hide nodes
)
nx.draw_networkx_edges(
    G,
    pos,
    arrowstyle='-|>',
    arrowsize=20,
    edge_color='gray',
    width=2
)
nx.draw_networkx_labels(
    G,
    pos,
    labels={node: node for node in G.nodes},
    font_size=12,
    font_weight='bold',
    bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='white')
)

# Add edge labels
edge_labels = {(edge[0], edge[1]): edge[2] for edge in edges}
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=10,
    font_color='blue',
    label_pos=0.5  # Position label in the center of the edge
)

# Title and display adjustments
plt.title("Experiment Workflow", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()
