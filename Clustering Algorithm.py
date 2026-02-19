#@title necessary import for second stage

from skimage.restoration import denoise_tv_chambolle
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, jensenshannon, cdist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,adjusted_rand_score,pairwise_distances,adjusted_rand_score,adjusted_mutual_info_score, v_measure_score, homogeneity_score, completeness_score, fowlkes_mallows_score
import itertools
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import convolve

#@title Dataframe Preparation (role based featuring)

def prepare_block_features_4(df, team_id):
    """
    Takes raw event dataframe and a team_id.
    Returns 'block_features' dataframe indexed 1-400 with max scaled
    TACTICAL ROLE ARCHETYPES.
    """

    df_processed = df.copy()
    columns = ['tag_0','tag_1','tag_2','tag_3','tag_4','tag_5']
    df_processed['Successful'] = (df_processed[columns] == 1801).any(axis=1).astype(int)

    tag_cols = [col for col in df_processed.columns if 'tags' in col]
    has_interception_tag = df_processed[tag_cols].isin([1401]).any(axis=1)
    is_not_clearance = df_processed['subEventName'] != 'Clearance'
    df_processed['interception'] = (has_interception_tag & is_not_clearance).astype(int)

    cols = ['teamId','eventId','eventName','subEventId','subEventName',
            'positions/0/y','positions/0/x','positions/1/y','positions/1/x',
            'tag_0','tag_1','tag_2','tag_3','tag_4','tag_5', 'interception']

    existing_cols = [c for c in cols if c in df_processed.columns]
    if 'Successful' in df_processed.columns:
        existing_cols.append('Successful')

    filtered_df = df_processed[existing_cols].copy()

    cond_1 = filtered_df['eventName'] == 'Pass'
    cond_2 = filtered_df['eventName'] == 'Shot'
    cond_3 = filtered_df['eventName'] == 'Duel'
    cond_4 = filtered_df['subEventName'] == 'Free kick cross'
    cond_5 = filtered_df['subEventName'] == 'Clearance'
    cond_6 = filtered_df['interception'] == 1

    filtered_df = filtered_df[cond_1 | cond_2 | cond_3 | cond_4 | cond_5 | cond_6].copy()

    filtered_df.loc[filtered_df['subEventName'].isin(['Cross', 'Free kick cross']), 'eventName'] = 'Cross'
    filtered_df.loc[filtered_df['subEventName'].isin(['Clearance']), 'eventName'] = 'Clearance'
    filtered_df.loc[filtered_df['subEventName'].isin(['Ground defending duel']), 'eventName'] = 'def_duels'
    filtered_df.loc[filtered_df['subEventName'].isin(['Ground attacking duel']), 'eventName'] = 'attack_duels'
    filtered_df.loc[filtered_df['subEventName'].isin(['Air duel']), 'eventName'] = 'air_duels'

    filtered_df['dx'] = filtered_df['positions/1/x'] - filtered_df['positions/0/x']
    filtered_df['dy'] = filtered_df['positions/1/y'] - filtered_df['positions/0/y']
    angle_rad = np.arctan2(filtered_df['dy'], filtered_df['dx'])
    filtered_df['angle_deg'] = (np.degrees(angle_rad) + 360) % 360
    filtered_df['pass_len'] = np.sqrt(filtered_df['dx']**2 + filtered_df['dy']**2)

    start_dist_center = abs(filtered_df['positions/0/y'] - 50)
    end_dist_center   = abs(filtered_df['positions/1/y'] - 50)


    df_team = filtered_df[filtered_df['teamId'] == team_id].copy()
    x_bin_start = df_team['positions/0/x'].clip(0,99) // 5
    y_bin_start = df_team['positions/0/y'].clip(0,99) // 5
    df_team['block_id'] = (x_bin_start * 20) + y_bin_start + 1


    is_pass = df_team['eventName'] == 'Pass'
    df_team['raw_pass_count'] = is_pass.astype(int)

    mask_fwd = (df_team['angle_deg'] >= 315) | (df_team['angle_deg'] <= 45)
    mask_bwd = (df_team['angle_deg'] >= 135) & (df_team['angle_deg'] <= 225)
    mask_lat = (~mask_fwd) & (~mask_bwd)
    mask_long  = df_team['pass_len'] > 25
    mask_short = df_team['pass_len'] < 10
    mask_lat_long = df_team['pass_len'] >= 18
    mask_lat_short = df_team['pass_len'] < 18
    mask_out = (end_dist_center > (start_dist_center + 3))
    mask_in  = (end_dist_center < (start_dist_center - 3))
    is_own_half = df_team['positions/0/x'] < 50
    is_opp_half = df_team['positions/0/x'] >= 50

    is_def_action = (df_team['interception'] == 1) | (df_team['eventName'] == 'Clearance') | (df_team['eventName'] == 'def_duels')
    df_team['role_fortress'] = (is_def_action & is_own_half).astype(int)
    df_team['role_press'] = (is_def_action & is_opp_half).astype(int)
    df_team['role_launch'] = (is_pass & mask_fwd & mask_long & is_own_half).astype(int)
    df_team['role_pivot'] = (is_pass & mask_lat & mask_out).astype(int)
    df_team['role_support'] = (is_pass & mask_lat & mask_in).astype(int)
    df_team['role_entry'] = (is_pass & mask_fwd & mask_short & is_opp_half).astype(int)

    is_winger_action = (df_team['eventName'] == 'attack_duels') | (df_team['eventName'] == 'Cross')
    df_team['role_isolation'] = is_winger_action.astype(int)
    df_team['role_finish'] = (df_team['eventName'] == 'Shot').astype(int)
    df_team['role_safety'] = (is_pass & mask_bwd).astype(int)
    df_team['role_switch'] = (is_pass & mask_lat & mask_lat_long).astype(int)
    df_team['role_link'] = (is_pass & mask_lat & mask_lat_short).astype(int)

    archetypes = [
        'raw_pass_count',
        'role_fortress', 'role_press',
        'role_launch', 'role_entry',
        'role_switch', 'role_link',
        'role_pivot', 'role_support',
        'role_isolation', 'role_finish',
        'role_safety'
    ]

    block_features = df_team.groupby('block_id')[archetypes].sum()

    all_blocks = range(1, 401)
    block_features_count = block_features.reindex(all_blocks, fill_value=0)

    max_vals = block_features_count.max().replace(0, 1)
    block_features_scaled = block_features_count.div(max_vals)

    return block_features_count, block_features_scaled

#@title Weighted denoiser

def denoise_custom_kernel(df_scaled,b=2, grid_size=(20, 20)):
    df_denoised = df_scaled.copy()

    weights = np.array([
        [1/(b**2), 1/(b**2), 1/(b**2), 1/(b**2), 1/(b**2)],
        [1/(b**2), 1/b,  1/b,  1/b,  1/(b**2)],
        [1/(b**2), 1/b,  1.0,  1/b,  1/(b**2)],
        [1/(b**2), 1/b,  1/b,  1/b,  1/(b**2)],
        [1/(b**2), 1/(b**2), 1/(b**2), 1/(b**2), 1/(b**2)]
    ])

    for col in df_scaled.columns:
        raw_grid = df_scaled[col].values.reshape(grid_size, order='F')
        smooth_grid = convolve(raw_grid, weights, mode='constant', cval=0.0)
        df_denoised[col] = smooth_grid.flatten(order='F')

    return df_denoised

#@title Calculate hellinger matrix

def calculate_hellinger_matrix(df, feature_cols):

    """
    Calculates the Pairwise Hellinger Distance Matrix.
    Formula: H(P, Q) = (1/sqrt(2)) * sqrt( sum( (sqrt(pi) - sqrt(qi))^2 ) )
    Range: [0, 1] (0 = Identical, 1 = Max Distance)
    """
    data = df[feature_cols].copy()
    row_sums = data.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    distributions = data.div(row_sums, axis=0).fillna(0)

    sqrt_data = np.sqrt(distributions.values)
    euclidean_dists = pdist(sqrt_data, metric='euclidean')
    hellinger_flat = euclidean_dists / np.sqrt(2)
    matrix_hellinger = squareform(hellinger_flat)
    df_hellinger = pd.DataFrame(
        matrix_hellinger,
        index=df.index,
        columns=df.index
    )

    return df_hellinger

#@title denoised flow matrix

_flow_denoising_weights = np.array([
    [0.25/3, 0.25/3, 0.25/3],
    [0.25/3, 1/3, 0.25/3],
    [0.25/3, 0.25/3, 0.25/3]
])

def calculate_and_plot_flow_denoised(df, team_id, min_actions=5, N_row=20, denoised=True):
    df = df.copy()
    df['dx'] = df['positions/1/x'] - df['positions/0/x']
    df['dy'] = df['positions/1/y'] - df['positions/0/y']
    filtered_df = df[(df['eventName'] == 'Pass') & (df['teamId'] == team_id)].copy()

    bin_width = 100 / N_row
    x_bin = filtered_df['positions/0/x'].clip(0, 99) // bin_width
    y_bin = filtered_df['positions/0/y'].clip(0, 99) // bin_width
    filtered_df['start'] = (x_bin * N_row) + y_bin + 1
    block_stats = filtered_df.groupby('start')[['dx', 'dy']].agg(['mean', 'count'])
    valid_mask = block_stats[('dx', 'count')] >= min_actions
    flow_vectors = block_stats[valid_mask].xs('mean', axis=1, level=1)

    total_blocks = N_row * N_row
    flow_vectors_full = flow_vectors.reindex(range(1, total_blocks + 1), fill_value=0)


    if denoised:
        dx_grid = flow_vectors_full['dx'].values.reshape(N_row, N_row)
        dy_grid = flow_vectors_full['dy'].values.reshape(N_row, N_row)

        dx_smoothed_grid = convolve(dx_grid, _flow_denoising_weights, mode='constant', cval=0.0)
        dy_smoothed_grid = convolve(dy_grid, _flow_denoising_weights, mode='constant', cval=0.0)
        final_vectors = np.stack([dx_smoothed_grid.flatten(), dy_smoothed_grid.flatten()], axis=1)
    else:
        final_vectors = flow_vectors_full.values

    flow_sim = cosine_similarity(final_vectors)
    flow_dist = (1 - flow_sim) / 2
    is_empty = (final_vectors[:, 0] == 0) & (final_vectors[:, 1] == 0)
    flow_dist[is_empty, :] = 0.5
    flow_dist[:, is_empty] = 0.5
    np.fill_diagonal(flow_dist, 0.0)

    return pd.DataFrame(flow_dist, index=range(1, total_blocks + 1), columns=range(1, total_blocks + 1))

#@title Hierarchical Clustering Function

def fuse_and_cluster(df_jsd, df_flow, alpha=0.5, dist_threshold = 0.3 , N_row = 20):

    jsd_vals = df_jsd.values
    flow_vals = df_flow.values

    final_matrix = (alpha * jsd_vals) + ((1 - alpha) * flow_vals)
    final_matrix = np.nan_to_num(final_matrix, nan=0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        final_matrix,
        xticklabels=N_row,
        yticklabels=N_row,
        cmap='viridis',
        cbar_kws={'label': f'Combined Distance (alpha={alpha})'}
    )
    plt.title(f"Fused Dissimilarity Matrix\n(Weight: {alpha*100:.0f}% Style, {(1-alpha)*100:.0f}% Flow)")
    plt.xlabel("Block ID")
    plt.ylabel("Block ID")
    plt.show()

    print(f"Running Agglomerative Clustering (distance={dist_threshold})...")

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        metric='precomputed',
        linkage='complete',
    )

    labels = model.fit_predict(final_matrix)

    print("Clustering Complete.")
    print(f"Total Unique Zones: {len(np.unique(labels))}")

    return labels, final_matrix

#@title Plotting Clustering Label

def plot_tactical_zones(labels, N_row=20, team_name="Team"):

    grid = np.array(labels).reshape((N_row, N_row), order='F')
    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(grid,linecolor= "white", annot= False, cmap='tab20', cbar_kws={'label': 'Cluster ID'})
    plt.title(f"Tactical Clusters: {team_name} (20x20 Grid)", fontsize=15)
    plt.xlabel("Pitch Length (X-axis)")
    plt.ylabel("Pitch Width (Y-axis)")
    plt.show()


#@title Clusterinh whole season data and calculating silhouette score

id = 1625                           #@param
N = 20
a = 0.67                            #@param
dist= 0.2                           #@param

data = df_events[df_events['teamId'] == id].copy()
metric_5 = [
        'role_fortress',
        'role_launch',
        'role_entry',
         'role_switch', 'role_link',
        'role_pivot', 'role_support',
        'role_isolation', 'role_finish',
        'role_safety'
    ]


block_features_count,block_features_scaled = prepare_block_features_4(data, id)
denoised = denoise_custom_kernel(block_features_scaled)
df_hellinger = calculate_hellinger_matrix(denoised,metric_5)
df_flow = calculate_and_plot_flow_denoised(data, id,N_row = N)
labels_3,final_matrix_3 = fuse_and_cluster(df_hellinger, df_flow, alpha=a, dist_threshold=dist , N_row = N)

plot_tactical_zones(labels_3, N_row=20, team_name="Team 1625")
silhouette_score(final_matrix, labels, metric='precomputed')