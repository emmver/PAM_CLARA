import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import psutil

class PAM:
  """
    Partitioning Around Medoids (PAM) clustering.

    PAM is a clustering algorithm that is robust to outliers and is used when
    a dataset contains noise and outliers. It uses medoids to represent clusters,
    which are the most centrally located objects in a cluster.

    Attributes:
      k (int): The number of clusters.
      medoids (np.ndarray): Array of indices of medoids after fitting the model.
      cluster_labels (np.ndarray): Array of cluster labels for each point in the dataset.
  """
  def __init__(self, df, k):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Number of clusters 'k' must be a positive integer.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if len(df.drop_duplicates()) < k:
            raise ValueError("Number of clusters 'k' cannot be greater than the number of unique data points.")
        if len(df) < k:
            raise ValueError("Number of clusters 'k' cannot be greater than the number of data points.")

        self.df = df
        self.k = k
        self.medoids = None
        self.cluster_labels = None
        self.medoids_history = []  # To store medoids at each step
        self.silhouette_scores = []  # To store silhouette scores
        self.colors = plt.cm.Dark2(np.linspace(0, 1, k))  # Initialize colors based on k fo rvisualization


  def fit(self, verbose=0):
    #verbose 0 --> No printing
      start_time = time.time()

      self.distance_matrix = self.compute_distance_matrix()
      self.medoids = self.build_phase(verbose)
      self.medoids = self.swap_phase(verbose)
      self.cluster_labels = self.assign_labels()
      end_time = time.time()
      end_mem = psutil.Process().memory_info().rss
      self.execution_time = end_time - start_time
      if verbose >= 1:
          print(f"Final Medoids: {self.medoids}")
          print('Wall Time:',self.execution_time)
      if verbose >= 2:
          self.log_silhouette_score()
      return self

  def compute_distance_matrix(self):
      n = len(self.df)
      distance_matrix = np.zeros((n, n))
      for i in range(n):
          for j in range(i + 1, n):
              distance = np.linalg.norm(self.df.iloc[i] - self.df.iloc[j])
              distance_matrix[i, j] = distance_matrix[j, i] = distance
      return distance_matrix

  def build_phase(self, verbose=False):
      n = len(self.distance_matrix)
      medoids = []

      # Find the global medoid
      total_dissimilarity = np.sum(self.distance_matrix, axis=1)
      global_medoid = np.argmin(total_dissimilarity)
      medoids.append(global_medoid)

      if verbose:
          print(f"Initial Medoid: {global_medoid}")

      # Iteratively select other medoids
      while len(medoids) < self.k:
          max_reduction = 0
          candidate_medoid = None

          for i in range(n):
              if i not in medoids:
                  reduction = 0
                  for j in range(n):
                      if j != i:
                          D_j = np.min([self.distance_matrix[j, m] for m in medoids])
                          d_ji = self.distance_matrix[j, i]
                          reduction += max(D_j - d_ji, 0)

                  if reduction > max_reduction:
                      max_reduction = reduction
                      candidate_medoid = i

          medoids.append(candidate_medoid)

          if verbose:
              print(f"Selected Medoid: {candidate_medoid}")

      self.medoids = np.array(medoids)
      self.medoids_history.append(self.medoids.copy())

      return self.medoids

  def total_cost(self, medoids):
      distances = self.distance_matrix[:, medoids]
      min_distances = np.min(distances, axis=1)
      return np.sum(min_distances)

  def swap_phase(self, verbose=False):
      n = len(self.distance_matrix)
      best_medoids = self.medoids.copy()
      best_cost = self.total_cost(self.medoids)
      for i in range(len(self.medoids)):
          for j in range(n):
              if j not in self.medoids:
                  new_medoids = self.medoids.copy()
                  new_medoids[i] = j
                  new_cost = self.total_cost(new_medoids)
                  if new_cost < best_cost:
                      best_cost = new_cost
                      best_medoids = new_medoids.copy()
                      self.medoids_history.append(best_medoids.copy())
                      self.silhouette_scores.append(silhouette_score(self.df, self.assign_labels()))

      if verbose:
          print(f"Medoids updated to: {best_medoids}")
      return best_medoids

  def assign_labels(self):
      distances_to_medoids = self.distance_matrix[:, self.medoids]
      return np.argmin(distances_to_medoids, axis=1)

  def evaluate_clustering_metrics(self):
    silhouette_avg = silhouette_score(self.df, self.cluster_labels)
    msd_sum = 0
    for cluster in np.unique(self.cluster_labels):
        cluster_points = self.df[self.cluster_labels == cluster]
        cluster_center = np.mean(cluster_points, axis=0)
        squared_distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
        msd_sum += np.sum(squared_distances) / len(cluster_points)
    msd_avg = msd_sum / len(np.unique(self.cluster_labels))
    return silhouette_avg, msd_avg

  def calculate_cluster_metrics(self):
    # Silhouette scores per cluster
    silhouette_vals = silhouette_samples(self.df, self.cluster_labels)
    self.cluster_silhouette_scores = {i: silhouette_vals[self.cluster_labels == i].mean() for i in range(self.k)}

    # MSD per cluster
    self.cluster_msd_scores = {}
    for cluster_idx in range(self.k):
        cluster_points = self.df[self.cluster_labels == cluster_idx]
        if len(cluster_points) > 1:
            cluster_center = self.df.iloc[self.medoids[cluster_idx]]
            squared_distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
            self.cluster_msd_scores[cluster_idx] = np.mean(squared_distances)

  def log_silhouette_score(self):
      score = silhouette_score(self.df, self.cluster_labels)
      print(f"Silhouette Score: {score}")

  def plot_silhouette(self, sample_size=None):
    if sample_size is None:
        df = self.df
        cluster_labels = self.cluster_labels
    else:
        random_indices = np.random.randint(0, self.df.shape[0], size=sample_size)
        df = self.df.iloc[random_indices]
        cluster_labels = self.cluster_labels[random_indices]

    silhouette_vals = silhouette_samples(df, cluster_labels)
    y_lower, y_upper = 0, 0
    unique_clusters = np.unique(cluster_labels)
    cluster_info = [(cluster, silhouette_vals[cluster_labels == cluster]) for cluster in unique_clusters]

    # Pre-sort silhouette values for each cluster
    for cluster, silhouette_vals in cluster_info:
        silhouette_vals.sort()
        y_upper += len(silhouette_vals)
        color = self.colors[cluster]
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_vals, facecolor=color, edgecolor=color)
        y_lower += len(silhouette_vals)

    # Plotting optimizations
    plt.axvline(x=silhouette_score(df, cluster_labels), color="red", linestyle="--")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient')
    plt.title('Silhouette Plot')
    plt.yticks([])  # Hide the y-axis ticks
    plt.show()

  def visualize_clusters(self, scale_factor=1,max_points_per_cluster=200,method='pca3'):
      dim = self.df.shape[1]
      cmap = ListedColormap(self.colors)
      if self.df.shape[0]<max_points_per_cluster*self.k:
        max_points_per_cluster=-1
      if method == 'pca3':
          if dim > 3:
              reduction = PCA(n_components=3)
              reduced_data = reduction.fit_transform(self.df)
          else:
              reduced_data = self.df.values
      else:
          if method == 'pca2':
              reduction = PCA(n_components=2)
          elif method == 'tsne':
              reduction = TSNE(n_components=3)
          else:
              raise ValueError("Method must be 'pca', 'tsne', or 'auto'.")
          reduced_data = reduction.fit_transform(self.df)

      if reduced_data.shape[1] == 3:
          views = [(0, 90), (0, -90), (0, 0), (0, 180), (90, 0), (-90, 0)]
          titles = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']

          fig = plt.figure(figsize=(15, 10))
          for i, view in enumerate(views):
              ax = fig.add_subplot(2, 3, i + 1, projection='3d')
              ax.view_init(elev=view[0], azim=view[1])
              for cluster in np.unique(self.cluster_labels):
                cluster_indices = np.where(self.cluster_labels == cluster)[0]
                np.random.shuffle(cluster_indices)
                sampled_indices = cluster_indices[:max_points_per_cluster]

                ax.scatter(reduced_data[sampled_indices, 0], reduced_data[sampled_indices, 1], reduced_data[sampled_indices, 2],
                           c=[self.colors[cluster]] * len(sampled_indices),
                           marker='o', edgecolor='k', s=30, alpha=0.7)
              ax.set_title(titles[i])
              ax.set_xlabel('Component 1')
              ax.set_ylabel('Component 2')
              ax.set_zlabel('Component 3')
      else:
        plt.figure(figsize=(10, 6))
        for cluster in np.unique(self.cluster_labels):
            cluster_indices = np.where(self.cluster_labels == cluster)[0]
            np.random.shuffle(cluster_indices)
            sampled_indices = cluster_indices[:max_points_per_cluster]

            cluster_points = reduced_data[sampled_indices]
            color = self.colors[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=color, marker='o', edgecolor='k', s=30, label=f'Cluster {cluster}')

            # Calculate covariance and plot ellipse
            cov_matrix = np.cov(cluster_points, rowvar=False)
            lambda_, v = np.linalg.eig(cov_matrix)
            lambda_ = np.sqrt(lambda_)*scale_factor
            ell = mpatches.Ellipse(xy=(np.mean(cluster_points, axis=0)),
                                    width=lambda_[0], height=lambda_[1],
                                    angle=np.rad2deg(np.arccos(v[0, 0])),
                                    edgecolor='black', facecolor=color,alpha=0.3)
            plt.gca().add_artist(ell)

            # Marking the medoid
            medoid_point = reduced_data[self.medoids[int(cluster)]]
            plt.scatter(medoid_point[0], medoid_point[1],
                        color=color, marker='x', edgecolor='k', s=100)

        plt.title('2D Cluster Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()

      #plt.show()

  def enrich_dataset(self):
      self.df['cluster_label'] = self.cluster_labels
      self.df['medoid'] = False
      self.df.loc[self.medoids, 'medoid'] = True
      return self.df

  def generate_report(self, scale_factor=1, return_report=False, save_markdown=False, save_plots=False, path='.',file_name="PAM_Report.md"):
      report = []

      report.append("## Medoids at Each Step\n")
      for step, medoids in enumerate(self.medoids_history):
          report.append(f"**Step {step + 1}:** {medoids}\n")

      report.append("\n## Silhouette Scores per Step\n")
      for step, score in enumerate(self.silhouette_scores):
          report.append(f"**Step {step + 1}:** {score}\n")

      # Add cluster metrics to the report
      self.calculate_cluster_metrics()
      report.append("\n## Cluster Silhouette Scores\n")
      for cluster, score in self.cluster_silhouette_scores.items():
          report.append(f"Cluster {cluster}: {score}\n")

      report.append("\n## Cluster MSD Scores\n")
      for cluster, score in self.cluster_msd_scores.items():
          report.append(f"Cluster {cluster}: {score}\n")

      report.append("\n## Final Medoids\n")
      report.append(f"{self.medoids}\n")

      report.append("\n## Performance Metrics\n")
      report.append(f"Execution Time: {self.execution_time} seconds\n")

      report_content = "\n".join(report)
      if return_report:
          return report_content

      if save_markdown:
          import os
          reports_folder = os.path.join(path, "reports")
          os.makedirs(reports_folder, exist_ok=True)
          with open(os.path.join(reports_folder, file_name), "w") as file:
              file.write(report_content)
          print(f"Report saved at {os.path.join(reports_folder, file_name)}")


      if save_plots:
            visuals_folder = os.path.join(path, "visualizations")
            os.makedirs(visuals_folder, exist_ok=True)

            # Save Silhouette Plot
            self.plot_silhouette()
            plt.savefig(os.path.join(visuals_folder, "Silhouette_Plot.png"))
            plt.close()

            # Save Cluster Visualization
            self.visualize_clusters(scale_factor=scale_factor)
            plt.savefig(os.path.join(visuals_folder, "Cluster_Visualization.png"))
            plt.close()