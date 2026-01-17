"""
Clustering Analysis Module
Identifies distinct student groups using various clustering algorithms
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class StudentClustering:
    """Performs clustering analysis to identify distinct student groups"""
    
    def __init__(self, n_clusters=3):
        """
        Initialize clustering analyzer
        
        Args:
            n_clusters: Number of clusters to identify (default: 3 for at-risk groups)
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        self.feature_columns = None
        
    def prepare_features(self, df, feature_columns=None):
        """
        Prepare features for clustering
        
        Args:
            df: DataFrame with student data
            feature_columns: List of columns to use for clustering
            
        Returns:
            Scaled feature matrix
        """
        if feature_columns is None:
            # Auto-select numeric columns related to stress and mental health
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter for relevant columns
            feature_columns = [col for col in numeric_cols 
                             if any(keyword in col.lower() for keyword in 
                                   ['stress', 'score', 'anxiety', 'depression', 'sleep', 'academic'])]
        
        self.feature_columns = feature_columns
        
        # Select and scale features
        X = df[feature_columns].copy()
        
        # Handle missing values - fill with median
        for col in X.columns:
            if X[col].isna().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def kmeans_clustering(self, X, n_clusters=None):
        """
        Perform K-means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (default: self.n_clusters)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels_ = self.model.fit_predict(X)
        
        return self.labels_
    
    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering for density-based grouping
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Cluster labels
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X)
        
        return self.labels_
    
    def hierarchical_clustering(self, X, n_clusters=None, linkage='ward'):
        """
        Perform hierarchical/agglomerative clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = self.model.fit_predict(X)
        
        return self.labels_
    
    def find_optimal_clusters(self, X, max_clusters=10, method='kmeans'):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to test
            method: Clustering method ('kmeans', 'hierarchical')
            
        Returns:
            Optimal number of clusters
        """
        silhouette_scores = []
        davies_bouldin_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            labels = model.fit_predict(X)
            
            if len(set(labels)) > 1:  # Need at least 2 clusters
                silhouette_scores.append(silhouette_score(X, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X, labels))
            else:
                silhouette_scores.append(-1)
                davies_bouldin_scores.append(float('inf'))
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def analyze_clusters(self, df, labels):
        """
        Analyze and describe each cluster
        
        Args:
            df: Original DataFrame
            labels: Cluster labels
            
        Returns:
            DataFrame with cluster statistics
        """
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # DBSCAN noise points
                continue
            
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_clustered) * 100
            }
            
            # Add mean values for numeric columns
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'cluster':
                    stats[f'{col}_mean'] = cluster_data[col].mean()
            
            cluster_stats.append(stats)
        
        cluster_df = pd.DataFrame(cluster_stats)
        return cluster_df
    
    def visualize_clusters(self, df, labels, feature_columns, save_path=None):
        """
        Visualize clustering results
        
        Args:
            df: Original DataFrame
            labels: Cluster labels
            feature_columns: Features used for clustering
            save_path: Path to save visualization
        """
        if len(feature_columns) < 2:
            print("Need at least 2 features for visualization")
            return
        
        # Use first two features for 2D visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        scatter = axes[0].scatter(
            df[feature_columns[0]], 
            df[feature_columns[1]], 
            c=labels, 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        axes[0].set_xlabel(feature_columns[0])
        axes[0].set_ylabel(feature_columns[1])
        axes[0].set_title('Student Clusters')
        plt.colorbar(scatter, ax=axes[0])
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[1].bar(unique_labels, counts, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Cluster ID')
        axes[1].set_ylabel('Number of Students')
        axes[1].set_title('Cluster Size Distribution')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def cluster_pipeline(self, df, feature_columns=None, method='kmeans', visualize=True):
        """
        Complete clustering pipeline
        
        Args:
            df: DataFrame with student data
            feature_columns: Columns to use for clustering
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            visualize: Whether to create visualizations
            
        Returns:
            DataFrame with cluster labels and cluster statistics
        """
        # Prepare features
        X, feature_cols = self.prepare_features(df, feature_columns)
        
        # Find optimal clusters if using kmeans or hierarchical
        if method in ['kmeans', 'hierarchical']:
            optimal_k = self.find_optimal_clusters(X, method=method)
            self.n_clusters = optimal_k
        
        # Perform clustering
        if method == 'kmeans':
            labels = self.kmeans_clustering(X)
        elif method == 'dbscan':
            labels = self.dbscan_clustering(X)
        elif method == 'hierarchical':
            labels = self.hierarchical_clustering(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Analyze clusters
        cluster_stats = self.analyze_clusters(df, labels)
        
        # Add labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        # Visualize if requested
        if visualize and len(feature_cols) >= 2:
            self.visualize_clusters(df, labels, feature_cols, 
                                   save_path='data/outputs/cluster_visualization.png')
        
        print(f"\nClustering complete!")
        print(f"Identified {len(set(labels))} distinct student groups")
        print("\nCluster Statistics:")
        print(cluster_stats.to_string())
        
        return df_clustered, cluster_stats


if __name__ == "__main__":
    # Example usage
    cluster_analyzer = StudentClustering(n_clusters=3)
    
    # Example with sample data
    # df = pd.read_csv('data/processed/processed_data.csv')
    # df_clustered, stats = cluster_analyzer.cluster_pipeline(df, method='kmeans')
    
    print("Clustering module ready!")
