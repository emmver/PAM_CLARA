# PAM Class Documentation

## Overview
`PAM` (Partitioning Around Medoids) is a clustering algorithm that is particularly robust to outliers. This class implementation provides methods for clustering datasets using the PAM algorithm, where clusters are represented by medoids, which are the most centrally located objects in a cluster.

## Attributes
- `df` (DataFrame): The dataset to be clustered.
- `k` (int): The number of clusters.
- `medoids` (np.ndarray): Array of indices of medoids after fitting the model.
- `cluster_labels` (np.ndarray): Array of cluster labels for each point in the dataset.

## Methods
### `__init__(self, df, k)`
Initialize the PAM instance.
- `df` (DataFrame): The dataset to be clustered.
- `k` (int): The number of clusters.

### `fit(self, verbose=0)`
Fit the PAM model to the dataset.
- `verbose` (int, optional): Level of verbosity. Defaults to 0 (silent).

### `compute_distance_matrix(self)`
Compute the distance matrix for the dataset.

### `build_phase(self, verbose=False)`
Build phase of the PAM algorithm.
- `verbose` (bool, optional): If True, print detailed logs. Defaults to False.

### `total_cost(self, medoids)`
Calculate the total cost of a given set of medoids.
- `medoids` (np.ndarray): Array of medoid indices.

### `swap_phase(self, verbose=False)`
Swap phase of the PAM algorithm.
- `verbose` (bool, optional): If True, print detailed logs. Defaults to False.

### `assign_labels(self)`
Assign labels to each point in the dataset based on the nearest medoid.

### `evaluate_clustering_metrics(self)`
Evaluate clustering using silhouette score and Mean Squared Distance (MSD).

### `calculate_cluster_metrics(self)`
Calculate silhouette scores and Mean Squared Distance (MSD) for each cluster.

### `log_silhouette_score(self)`
Log the silhouette score of the current clustering.

### `plot_silhouette(self, sample_size=None)`
Plot the silhouette for the current clustering.
- `sample_size` (int, optional): Number of samples to plot. If None, use the entire dataset.

### `visualize_clusters(self, scale_factor=1, max_points_per_cluster=200, method='pca3')`
Visualize clusters in 2D or 3D.
- `scale_factor` (float, optional): Scale factor for the plot size. Defaults to 1.
- `max_points_per_cluster` (int, optional): Maximum number of points to plot per cluster. Defaults to 200.
- `method` (str, optional): Method for dimensionality reduction ('pca3', 'pca2', 'tsne'). Defaults to 'pca3'.

### `enrich_dataset(self)`
Enrich the original dataset with cluster labels and medoid flags.

### `generate_report(self, return_report=False, save_markdown=False, path='.', file_name="PAM_Report.md")`
Generate a comprehensive report of the PAM clustering process.
- `return_report` (bool, optional): If True, return the report as a string. Defaults to False.
- `save_markdown` (bool, optional): If True, save the report as a Markdown file. Defaults to False.
- `path` (str, optional): Path where the markdown file will be saved. Defaults to the current directory.
- `file_name` (str, optional): Name of the markdown file. Defaults to "PAM_Report.md".

