# CLARA Class Documentation

## Overview
CLARA (Clustering LARge Applications) is an implementation of k-medoids clustering optimized for large datasets. It applies the PAM (Partitioning Around Medoids) algorithm on smaller samples of the dataset and uses the best clustering result.

## Attributes
- `df` (DataFrame): The dataset for clustering.
- `k` (int): The number of clusters.
- `num_samples` (int): Number of samples to draw from the dataset.
- `sample_size` (int): The size of each sample, defaults to 40 + 2 * k.
- `best_medoids` (list): Best medoids found during fitting.
- `best_score` (float): Best silhouette score obtained.
- `cluster_labels` (list): Cluster labels for each point in the dataset.

## Methods
### `__init__(self, df, k, num_samples=5, sample_size=None)`
Initialize the CLARA instance.
- `df` (DataFrame): The dataset to be clustered.
- `k` (int): The number of clusters.
- `num_samples` (int, optional): The number of samples to use. Defaults to 5.
- `sample_size` (int, optional): The size of each sample. Defaults to 40 + 2 * k.

### `_construct_sample(self, include_medoids=None)`
Construct a sample from the dataset, ensuring no duplicates and including the best medoids if provided.
- `include_medoids` (list, optional): Medoids to include in the sample.

### `fit(self, verbose=0)`
Fit the CLARA model to the dataset.
- `verbose` (int, optional): Level of verbosity. Defaults to 0 (silent).

### `average_distance_to_medoid(self, medoids, labels)`
Compute the average distance of all points to their nearest medoid.
- `medoids` (list): The indices of the medoids.
- `labels` (list): The cluster labels for each point in the dataset.

### `evaluate_clustering_metrics(self, data, labels)`
Evaluate clustering performance using silhouette score and Mean Squared Distance (MSD).
- `data` (DataFrame): The dataset or a sample of it.
- `labels` (list): The cluster labels for the dataset or sample.

### `assign_labels_to_full_data(self, medoids)`
Assign cluster labels to the full dataset based on the provided medoids.
- `medoids` (list): The indices of the medoids.

### `plot_silhouette(self, sample_size)`
Plot the silhouette for the best clustering found.
- `sample_size` (int): The size of the sample to be used for silhouette plotting.

### `visualize_clusters(self, scale_factor=1, max_points_per_cluster=200, method='pca3')`
Visualize clusters for the best clustering found.
- `scale_factor` (float, optional): Scale factor for the plot size. Defaults to 1.
- `max_points_per_cluster` (int, optional): Maximum number of points to plot per cluster. Defaults to 200.
- `method` (str, optional): Dimensionality reduction method ('pca3' for 3D PCA, 'tsne' for t-SNE). Defaults to 'pca3'.

### `enrich_dataset(self)`
Enrich the original dataset with cluster labels and medoid flags.

### `generate_clara_report(self, file_name='CLARA_Report.md',save_markdown=False, path='.')`
Generate a report for the CLARA clustering process.
- `file_name` (str, optional): Name of the markdown file to be generated. Defaults to 'CLARA_Report.md'.
- `save_markdown` (bool, optional): Whether to save the report as a markdown file. Defaults to False.
- `path` (str, optional): Path where the markdown file will be saved. Defaults to the current directory.
