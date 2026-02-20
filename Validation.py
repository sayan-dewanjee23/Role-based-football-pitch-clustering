#@title parameter validation

import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd


alpha_range = np.linspace(0.3, 0.7, 11)
b_range = [1,2, 3, 4, 5]
data = df_events[df_events["teamId"]==1625].copy()

results = []
best_score = -1
best_params = {}


for b in b_range:
    block_features_count, block_features_scaled = prepare_block_features_4(data, id)
    denoised = denoise_custom_kernel(block_features_scaled, b=b)

    df_hellinger = calculate_hellinger_matrix(denoised, metric_5)
    df_flow = calculate_and_plot_flow_denoised(data, id, N_row=N)

    for a in alpha_range:
        try:
            labels, final_matrix = fuse_and_cluster(
                df_hellinger,
                df_flow,
                alpha=a,
                dist_threshold=dist,
                N_row=N
            )

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if 1 < n_clusters < len(final_matrix):
                score = silhouette_score(final_matrix, labels)

                results.append({'alpha': a, 'b': b, 'score': score, 'clusters': n_clusters})

                if score > best_score:
                    best_score = score
                    best_params = {'alpha': a, 'b': b, 'score': score}
            else:
                results.append({'alpha': a, 'b': b, 'score': np.nan, 'clusters': n_clusters})

        except Exception as e:
            print(f"Error at a={a}, b={b}: {e}")
            continue

df_results = pd.DataFrame(results)
print("Best Parameters Found:")
print(best_params)


# 1. Team IDs ordered by League Position (Man City, Man Utd, Tottenham, etc.)
team_ids = [1625, 1611, 1624, 1612, 1610, 1609, 1646, 1623, 1631, 1613,
            1628, 1659, 1633, 1644, 1651, 1673, 1619, 10531, 1639, 1627]


#@title Silhouette score plot
metric_5 = [
    'role_fortress', 'role_launch', 'role_entry', 'role_switch', 'role_link',
    'role_pivot', 'role_support', 'role_isolation', 'role_finish', 'role_safety'
]

team_labels_for_plot = []
silhouette_scores = []

print("Starting clustering loop for all 20 teams...")

for tid in team_ids:
    data = df_events[df_events['teamId'] == tid].copy()

    block_features_count, block_features_scaled = prepare_block_features_4(data, tid)
    denoised = denoise_custom_kernel(block_features_scaled)
    df_hellinger = calculate_hellinger_matrix(denoised, metric_5)
    df_flow = calculate_and_plot_flow_denoised(data, tid, N_row=N)

    labels, final_matrix = fuse_and_cluster(df_hellinger, df_flow, alpha=a, dist_threshold=dist, N_row=N)

    num_clusters = len(np.unique(labels))
    if 1 < num_clusters < len(labels):
        score = silhouette_score(final_matrix, labels, metric='precomputed')

        silhouette_scores.append(score)
        team_labels_for_plot.append(str(tid))  # Convert to string so x-axis treats them as categories

        print(f"Team {tid} | Clusters: {num_clusters} | Silhouette Score: {score:.4f}")
    else:
        print(f"Team {tid} | Failed to cluster properly (Found {num_clusters} clusters).")

plt.figure(figsize=(12,6))

sns.pointplot(x=team_labels_for_plot, y=silhouette_scores,
              color='#1f77b4', scale=1.2, markers='o')

plt.title("Tactical Structural Stability Across Premier League Teams (17/18)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Team ID (Ordered by Final League Position)", fontsize=12, labelpad=10)
plt.ylabel("Silhouette Score (Cluster Cohesion & Separation)", fontsize=12, labelpad=10)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.xticks(rotation=45)
plt.ylim(0.22, max(silhouette_scores) + 0.1) # Give a little headroom above the highest point

plt.tight_layout()
plt.show()


#@title label permutation test

def run_permutation_test(final_matrix, actual_labels, n_permutations=1000):
    """
    Permutes cluster labels to build a null distribution of Silhouette Scores.
    """
    actual_score = silhouette_score(final_matrix, actual_labels, metric='precomputed')
    null_scores = []

    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(actual_labels)
        score = silhouette_score(final_matrix, shuffled_labels, metric='precomputed')
        null_scores.append(score)

    null_scores = np.array(null_scores)

    p_value = np.sum(null_scores >= actual_score) / n_permutations

    return actual_score, null_scores, p_value

def plot_permutation_results(null_mci, actual_mci, p_mci, null_wba, actual_wba, p_wba):
    """
    Plots the null distributions and actual scores for both teams on one axis.
    """
    plt.figure(figsize=(12, 7))

    sns.kdeplot(null_mci, fill=True, color='skyblue', alpha=0.5, label='Man City (Null)')
    plt.axvline(actual_mci, color='blue', linestyle='-', linewidth=2,
                label=f'Man City Actual (p < 0.01)' if p_mci < 0.01 else f'Man City Actual (p={p_mci:.3f})')

    sns.kdeplot(null_wba, fill=True, color='lightcoral', alpha=0.5, label='West Brom (Null)')
    plt.axvline(actual_wba, color='red', linestyle='-', linewidth=2,
                label=f'West Brom Actual (p < 0.01)' if p_wba < 0.01 else f'West Brom Actual (p={p_wba:.3f})')

    plt.title("Label Permutation Test: Validating Cluster Significance Against Randomness", fontsize=15, fontweight='bold')
    plt.xlabel("Silhouette Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    plt.text(0.05, 0.95, "Significance Threshold: p < 0.01",
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    metric_5 = [
        'role_fortress',
        'role_launch',
        #'role_press',
        'role_entry',
         'role_switch', 'role_link',
        'role_pivot', 'role_support',
        'role_isolation', 'role_finish',
        'role_safety'
    ]

data = df_events[df_events['teamId'] == 1625].copy()
block_features_count,block_features_scaled = prepare_block_features_4(data, 1625)
denoised = denoise_custom_kernel(block_features_scaled)
df_hellinger = calculate_hellinger_matrix(denoised,metric_5)
df_flow = calculate_and_plot_flow_denoised(data, 1625,N_row = N)
labels_3,final_matrix_3 = fuse_and_cluster(df_hellinger, df_flow, alpha=0.67, dist_threshold=0.2 , N_row = 20)
actual_mci, null_mci, p_mci = run_permutation_test(final_matrix_3, labels_3)

data = df_events[df_events['teamId'] == 1627].copy()
block_features_count,block_features_scaled = prepare_block_features_4(data, 1627)
denoised = denoise_custom_kernel(block_features_scaled)
df_hellinger = calculate_hellinger_matrix(denoised,metric_5)
df_flow = calculate_and_plot_flow_denoised(data, 1627,N_row = N)
labels_3,final_matrix_3 = fuse_and_cluster(df_hellinger, df_flow, alpha=0.67, dist_threshold=0.2 , N_row = 20)
actual_wba, null_wba, p_wba = run_permutation_test(final_matrix_3, labels_3)

plot_permutation_results(null_mci, actual_mci, p_mci, null_wba, actual_wba, p_wba)

#@title Heatmap test

team_id = 1610                                 #@param
metrics = [
        'role_fortress',
        'role_launch',
        'role_press',
        'role_entry',
         'role_switch', 'role_link',
        'role_pivot', 'role_support',
        'role_isolation', 'role_finish',
        'role_safety'
    ]

data = df_events[df_events["teamId"] == team_id].copy()
x_bin_start = data['positions/0/x'].clip(0,99) // 5
y_bin_start = data['positions/0/y'].clip(0,99) // 5
data['block_id'] = (x_bin_start * 20) + y_bin_start + 1


team_data = block_features_scaled[metrics].copy()

row_sums = team_data.sum(axis=1)
row_sums[row_sums == 0] = 1.0
team_prob_data = team_data.div(row_sums, axis=0).fillna(0)
team_prob_data["label"] = labels_3

cluster_profiles = team_prob_data.groupby('label').mean()

label_lookup = team_prob_data['label'].to_dict()
data['cluster_label'] = data['block_id'].map(label_lookup)

ownership_counts = data.groupby(['cluster_label', 'playerId']).size().unstack(fill_value=0)
cluster_ownership = ownership_counts.div(ownership_counts.sum(axis=1), axis=0)


top_5_summary = {}

for cluster_id in cluster_ownership.index:
    top_players = cluster_ownership.loc[cluster_id].nlargest(5)

    player_strings = [f"{name} ({prob*100:.1f}%)"
                      for name, prob in top_players.items()]

    top_5_summary[cluster_id] = "\n              ".join(player_strings)

aguero_profile = cluster_ownership[31528]

all_cluster_labels = np.unique(labels_3)
aguero_profile = aguero_profile.reindex(all_cluster_labels, fill_value=0)

heat_values = np.zeros(400)
for block_id, cluster_label in label_lookup.items():
    if 0 <= block_id -1 < 400: # Adjust block_id to be 0-indexed for array access
        heat_values[block_id - 1] = aguero_profile[cluster_label]

heat_matrix = heat_values.reshape((20, 20), order='F')

plt.figure(figsize=(10, 8))
sns.heatmap(heat_matrix, cmap="OrRd", annot=False)
plt.title("Tactical Heatmap: Kante (2017/18)")
plt.show()

#@title Graph building

import networkx as nx
import numpy as np 

team_id = 1646                                        #@param
num_cluster = len(np.unique(labels_3))

data = df_events[df_events["teamId"] == team_id].copy()
data = data[(data["eventName"]=="Pass")].copy()

x_bin_start = data['positions/0/x'].clip(0,99) // 5
y_bin_start = data['positions/0/y'].clip(0,99) // 5
x_bin_end = data['positions/1/x'].clip(0,99) // 5
y_bin_end = data['positions/1/y'].clip(0,99) // 5

data['start'] = (x_bin_start * 20) + y_bin_start + 1
data['end'] = (x_bin_end * 20) + y_bin_end + 1

data['start_cluster'] = data['start'].map(label_lookup)
data['end_cluster'] = data['end'].map(label_lookup)

edges = data.groupby(['start_cluster', 'end_cluster']).size().reset_index(name='weight')

G = nx.DiGraph()
G.add_nodes_from(np.unique(labels_3)) 

for _, row in edges.iterrows():
    w = row['weight']
    dist = 1.0 / w if w > 0 else 0
    G.add_edge(row['start_cluster'], row['end_cluster'], weight=w, distance=dist)

import networkx as nx
import numpy as np 

team_id = 1646                                        #@param
num_cluster = len(np.unique(labels_3)) # This will be 39 (clusters 0-38)

data = df_events[df_events["teamId"] == team_id].copy()
data = data[(data["eventName"]=="Pass")].copy()

x_bin_start = data['positions/0/x'].clip(0,99) // 5
y_bin_start = data['positions/0/y'].clip(0,99) // 5
x_bin_end = data['positions/1/x'].clip(0,99) // 5
y_bin_end = data['positions/1/y'].clip(0,99) // 5

data['start'] = (x_bin_start * 20) + y_bin_start + 1
data['end'] = (x_bin_end * 20) + y_bin_end + 1

data['start_cluster'] = data['start'].map(label_lookup)
data['end_cluster'] = data['end'].map(label_lookup)

edges = data.groupby(['start_cluster', 'end_cluster']).size().reset_index(name='weight')

G = nx.DiGraph()
G.add_nodes_from(np.unique(labels_3)) # Add all unique cluster labels as nodes

for _, row in edges.iterrows():
    w = row['weight']
    dist = 1.0 / w if w > 0 else 0
    G.add_edge(row['start_cluster'], row['end_cluster'], weight=w, distance=dist)


cluster_coords_raw = data.groupby('start_cluster')[['positions/0/x', 'positions/0/y']].mean()

pos = {}
default_x = 97
default_y = 50
overlap_offset_x = 2 # Small offset to prevent direct overlap
overlap_offset_y = 2 # Small offset to prevent direct overlap
occupied_default_positions = 0

for i in range(num_cluster): # Iterate through all expected cluster IDs
    if i in cluster_coords_raw.index:
        pos[i] = (cluster_coords_raw.loc[i, 'positions/0/x'], cluster_coords_raw.loc[i, 'positions/0/y'])
    else:
        pos[i] = (default_x + (occupied_default_positions * overlap_offset_x),
                  default_y + (occupied_default_positions * overlap_offset_y))
        occupied_default_positions += 1

plt.figure(figsize=(14, 9))
plt.title("Man City 17/18: Tactical Flow Graph (Clusters 0-38)", fontsize=15)

threshold = 5                        #@param
large_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
weights = [G[u][v]['weight'] for u, v in large_edges]

max_weight = max(weights) if weights else 1
normalized_widths = [(w / max_weight) * 5 for w in weights]

nodes_collection = nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

edges_collection = nx.draw_networkx_edges(G, pos, edgelist=large_edges, width=normalized_widths,
                       arrowstyle='->', arrowsize=25, edge_color='gray', alpha=0.6)

all_legend_handles = []
all_legend_labels = []

source_node = 5   #@param
target_node = 5  #@param

try:
    path_red = nx.shortest_path(G, source=source_node, target=target_node, weight='distance')
    direct_path = nx.shortest_path_length(G, source=source_node, target=target_node, weight='distance')
    path_edges_red = list(zip(path_red, path_red[1:]))
    print(f"Calculated Path: {path_red}")

    nx.draw_networkx_nodes(G, pos, nodelist=path_red, node_size=600, node_color='red', edgecolors='black')

    path_line_red = nx.draw_networkx_edges(G, pos, edgelist=path_edges_red, width=4, edge_color='red',
                               arrowstyle='->', arrowsize=25)
    if path_line_red: all_legend_handles.append(path_line_red[0]); all_legend_labels.append('Path to Target 5')

    path_labels_red = {node: node for node in path_red}
    nx.draw_networkx_labels(G, pos, labels=path_labels_red, font_size=12, font_color='white', font_weight='bold')
except nx.NetworkXNoPath:
    print(f"No path found between {source_node} and {target_node}. Check your edge threshold.")

try:
    path_black = nx.shortest_path(G, source=source_node, target= 35, weight='distance')
    path_edges_black = list(zip(path_black, path_black[1:]))
    print(f"Calculated Path: {path_black}")

    nx.draw_networkx_nodes(G, pos, nodelist=path_black, node_size=600, node_color='red', edgecolors='black')
    path_line_black = nx.draw_networkx_edges(G, pos, edgelist=path_edges_black, width=4, edge_color='black',
                                arrowstyle='->', arrowsize=25)
    if path_line_black: all_legend_handles.append(path_line_black[0]); all_legend_labels.append('Path to Target 9')

    path_labels_black = {node: node for node in path_black}
    nx.draw_networkx_labels(G, pos, labels=path_labels_black, font_size=12, font_color='white', font_weight='bold')
except nx.NetworkXNoPath:
    print(f"No path found between {source_node} and 9. Check your edge threshold.")

# Path to target 35
try:
    path_green = nx.shortest_path(G, source=source_node, target=9, weight='distance')
    path_edges_green = list(zip(path_green, path_green[1:]))
    print(f"Calculated Path: {path_green}")

    nx.draw_networkx_nodes(G, pos, nodelist=path_green, node_size=600, node_color='red', edgecolors='black')
    path_line_green = nx.draw_networkx_edges(G, pos, edgelist=path_edges_green, width=4, edge_color='green',
                                 arrowstyle='->', arrowsize=25)
    if path_line_green: all_legend_handles.append(path_line_green[0]); all_legend_labels.append('Path to Target 35')

    path_labels_green = {node: node for node in path_green}
    nx.draw_networkx_labels(G, pos, labels=path_labels_green, font_size=12, font_color='white', font_weight='bold')
except nx.NetworkXNoPath:
    print(f"No path found between {source_node} and 35. Check your edge threshold.")

# Path to target 27
try:
    path_violet = nx.shortest_path(G, source=source_node, target= 27, weight='distance')
    path_edges_violet = list(zip(path_violet, path_violet[1:]))
    print(f"Calculated Path: {path_violet}")

    nx.draw_networkx_nodes(G, pos, nodelist=path_violet, node_size=600, node_color='red', edgecolors='black')
    path_line_violet = nx.draw_networkx_edges(G, pos, edgelist=path_edges_violet, width=4, edge_color='violet',
                                 arrowstyle='->', arrowsize=25)
    if path_line_violet: all_legend_handles.append(path_line_violet[0]); all_legend_labels.append('Path to Target 27')

    path_labels_violet = {node: node for node in path_violet}
    nx.draw_networkx_labels(G, pos, labels=path_labels_violet, font_size=12, font_color='white', font_weight='bold')
except nx.NetworkXNoPath:
    print(f"No path found between {source_node} and 27. Check your edge threshold.")


plt.axvline(x=50, color='black', linestyle='--', alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.gca().invert_yaxis()
plt.title(f"Tactical Highways from Cluster {source_node}", fontsize=15)
plt.legend(handles=all_legend_handles, labels=all_legend_labels, loc='upper left')
plt.show()

print(f"Direct Path Tactical Distance: {direct_path}")

#@title betweenness

betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True)

sorted_betweenness = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)

print("Top 5 Tactical Bridge Clusters:")
for cluster, score in sorted_betweenness[:5]:
    print(f"Cluster {cluster}: {score:.4f}")