import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Simulate example data (time, voltage, current, temperature, SOC)
np.random.seed(42)
n_samples = 100

time = np.linspace(0, 100, n_samples)
voltage = np.random.uniform(3, 5, n_samples)  # Voltage between 3 and 5
current = np.random.uniform(-5, 5, n_samples)  # Current between -5 and 5
temperature = np.random.uniform(0, 40, n_samples)  # Temperature between 0 and 40
soc = np.random.uniform(0, 100, n_samples)  # SOC between 0 and 100

# Define bins
voltage_bins = np.linspace(3, 5, 11)
current_bins = np.linspace(-5, 5, 11)
temperature_bins = np.linspace(0, 40, 11)
soc_bins = np.linspace(0, 100, 11)

# Initialize a graph to represent transitions
G = nx.DiGraph()

# Example function to dynamically add transitions to the graph
def update_graph_with_transition(graph, start_bin, end_bin, direction):
    edge = (start_bin, end_bin)
    if graph.has_edge(*edge):
        graph[edge[0]][edge[1]]['weight'] += 1
    else:
        graph.add_edge(start_bin, end_bin, weight=1, direction=direction)

# Iterate through the data stream and update the graph
for i in range(len(time) - 1):
    v_bin_current = np.digitize(voltage[i], voltage_bins) - 1
    v_bin_next = np.digitize(voltage[i+1], voltage_bins) - 1
    c_bin_current = np.digitize(current[i], current_bins) - 1
    c_bin_next = np.digitize(current[i+1], current_bins) - 1
    t_bin_current = np.digitize(temperature[i], temperature_bins) - 1
    t_bin_next = np.digitize(temperature[i+1], temperature_bins) - 1
    s_bin_current = np.digitize(soc[i], soc_bins) - 1
    s_bin_next = np.digitize(soc[i+1], soc_bins) - 1

    if soc[i+1] > soc[i]:
        direction = "charging"
    elif soc[i+1] < soc[i]:
        direction = "discharging"
    else:
        direction = "no change"

    # Update the graph for each transition
    update_graph_with_transition(G, (v_bin_current, c_bin_current, t_bin_current, s_bin_current),
                                     (v_bin_next, c_bin_next, t_bin_next, s_bin_next),
                                     direction)

# Correcting the visualization by removing the incorrect argument for edges

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)

# Drawing nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# Drawing edges with weights
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=[G[u][v]['weight'] for u, v in G.edges()], alpha=0.6)

# Drawing labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title("Dynamic Transition Graph Between Bins")
plt.show()

