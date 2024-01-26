import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import psutil
from PAM import * 
import random

class CLARA:
    def __init__(self, df, k, num_samples=5, sample_size=None):
        """
        Initialize the CLARA instance.
        """
        self.df = df
        self.k = k
        self.num_samples = num_samples
        if sample_size:
          self.sample_size = sample_size
        else:
          self.sample_size=40 + 2*k
        if self.sample_size > len(df):
          raise ValueError('Datapoints not enough, please use PAM')
        self.best_medoids = None
        self.best_score = float('inf')
        self.cluster_labels = None

    def _construct_sample(self, include_medoids=None):
        """
        Construct a sample, ensuring no duplicates and including best medoids if provided.
        """
        if include_medoids is not None:
            sample = set(include_medoids)
        else:
            sample = set()

        while len(sample) < self.sample_size:
            candidate = random.choice(range(len(self.df)))
            sample.add(candidate)
        selected_indices = list(sample)
        sample_df = self.df.iloc[selected_indices]
        return sample_df, selected_indices

    def fit(self, verbose=0):
        """
        Fit the CLARA model to the dataset.
        """
        start_time = time.time()

        self.pam_reports = []  # Store individual PAM step reports

        for sample_index in range(self.num_samples):
            if verbose >= 1:
                print(f"Processing sample {sample_index + 1}/{self.num_samples}...")
            
            # Construct the sample
            sample, selected_indices = self._construct_sample(include_medoids=self.best_medoids)
            pam = PAM(sample, self.k)
            pam.fit()
            sample_medoids = pam.medoids

            # Map sample medoids to original DataFrame indices
            original_medoids = [selected_indices[medoid] for medoid in sample_medoids]

            full_data_labels = self.assign_labels_to_full_data(original_medoids)
            average_distance = self.average_distance_to_medoid(original_medoids, full_data_labels)
            
            if average_distance < self.best_score:
                self.best_score = average_distance
                self.best_medoids = original_medoids  # Update with original DataFrame indices
                self.cluster_labels = full_data_labels
                if verbose >= 1:
                    print('Found better sample...')
                    print('Score:', self.best_score)
                    print('Medoids:', self.best_medoids)
            self.pam_reports.append(pam.generate_report(return_report=True))
        self.sample_silhouette_avg, self.sample_msd_avg = self.evaluate_clustering_metrics(sample, pam.cluster_labels)
        end_time = time.time()
        self.execution_time = end_time - start_time
        if verbose >= 1:
            print(f"\nBest score: {self.best_score}")
            print(f"Best medoids: {self.best_medoids}")
            print('Wall time:',self.execution_time)
        return self

    def average_distance_to_medoid(self, medoids, labels):
        """
        Compute the average distance of all points to their nearest medoid in a vectorized manner.
        """
        
        # Extract the points corresponding to the medoids
        medoid_points = self.df.iloc[medoids].values

        # Compute distances from each point to each medoid
        distances = np.linalg.norm(self.df.values[:, np.newaxis, :] - medoid_points, axis=2)

        # Use labels to index these distances
        nearest_medoid_distances = distances[np.arange(len(self.df)), labels.astype(int)]

        # Calculate the total distance and average it
        total_distance = np.sum(nearest_medoid_distances)
        average_distance = total_distance / len(self.df)

        return average_distance

    def evaluate_clustering_metrics(self, data, labels):
      """
      Evaluate clustering using silhouette score and internal cluster Mean Squared Distance (MSD).
      """
      silhouette_avg = silhouette_score(data, labels)

      msd_sum = 0
      for cluster in range(self.k):
          cluster_points = data[labels == cluster]
          if len(cluster_points) > 1:
              cluster_center = np.mean(cluster_points, axis=0)
              squared_distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
              msd_sum += np.sum(squared_distances) / len(cluster_points)

      msd_avg = msd_sum / self.k

      return silhouette_avg, msd_avg

    def assign_labels_to_full_data(self, medoids):
        """
        Assign labels to the full dataset based on the given medoids in a vectorized manner.
        """

        # Extract the points corresponding to the medoids
        medoid_points = self.df.iloc[medoids].values

        # Compute distances from each point to each medoid
        # Reshape the self.df to (-1, 1, number of features) to broadcast over medoids
        # Then compute the norm along the last axis (features axis)
        distances = np.linalg.norm(self.df.values[:, np.newaxis, :] - medoid_points, axis=2)

        # Find the index of the minimum distance for each point
        labels = np.argmin(distances, axis=1)

        return labels


    def plot_silhouette(self,sample_size):
        """
        Plot silhouette for the best clustering found.
        """
        if self.best_medoids is not None:
            pam = PAM(self.df, self.k)
            pam.medoids = self.best_medoids
            pam.cluster_labels = self.cluster_labels
            pam.plot_silhouette(sample_size=sample_size)
        else:
            print("No clustering performed yet.")

    def visualize_clusters(self, scale_factor=1, max_points_per_cluster=200,method='pca3'):
        """
        Visualize clusters for the best clustering found.
        """
        if self.best_medoids is not None:
            pam = PAM(self.df, self.k)
            pam.medoids = self.best_medoids
            pam.cluster_labels = self.cluster_labels
            pam.k=self.k
            pam.visualize_clusters(scale_factor=scale_factor, method=method, max_points_per_cluster=max_points_per_cluster)
        else:
            print("No clustering performed yet.")

    def enrich_dataset(self):
        """
        Enrich the original dataset with cluster labels and medoid flags.
        """
        if self.cluster_labels is not None:
            self.df['cluster_label'] = self.cluster_labels
            self.df['medoid'] = False
            for medoid_index in self.best_medoids:
              self.df.at[medoid_index, 'medoid'] = True
            return self.df
        else:
            print("No clustering performed yet.")
            return self.df

    def generate_clara_report(self, file_name='CLARA_Report.md',save_markdown=False, path='.'):
      clara_report = ["# CLARA Clustering Report\n"]
      clara_report.append(f"Best silhouette score: {self.best_score}\n")
      clara_report.append(f"Best medoids: {self.best_medoids}\n")

      for i, report_string in enumerate(self.pam_reports):
          report_lines = report_string.split('\n')  # Split the string into a list of lines
          clara_report.append(f"\n## Report for Sample {i + 1}\n")
          clara_report.extend(report_lines)  # Extend with the list of lines

      clara_report.append("\n## Performance Metrics\n")
      clara_report.append(f"Execution Time: {self.execution_time} seconds\n")

      full_report_content = "\n".join(clara_report)

      if save_markdown:
          import os
          reports_folder = os.path.join(path, "CLARA_reports")
          os.makedirs(reports_folder, exist_ok=True)
          with open(os.path.join(reports_folder, file_name), "w") as file:
              file.write(full_report_content)
          print(f"Report saved at {os.path.join(path, 'CLARA_reports', file_name)}")
