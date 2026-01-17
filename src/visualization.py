"""
Visualization Module
Creates comprehensive visualizations for dashboard and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DashboardVisualizer:
    """Creates visualizations for interactive dashboard"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.figures = {}
        
    def create_stress_timeline(self, df, date_column=None, stress_column=None, 
                              save_path=None, interactive=True):
        """
        Create timeline visualization of stress levels
        
        Args:
            df: DataFrame with temporal stress data
            date_column: Date/week column name
            stress_column: Stress indicator column
            save_path: Path to save figure
            interactive: Whether to create interactive Plotly chart
        """
        if date_column is None:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
            date_column = date_cols[0] if date_cols else None
        
        if stress_column is None:
            stress_cols = [col for col in df.columns if 'stress' in col.lower()]
            stress_column = stress_cols[0] if stress_cols else None
        
        if date_column is None or stress_column is None:
            raise ValueError("Date and stress columns must be specified")
        
        if interactive:
            fig = px.line(df, x=date_column, y=stress_column,
                         title='Stress Levels Over Time',
                         labels={stress_column: 'Stress Level', date_column: 'Time Period'})
            fig.update_traces(line_width=3, marker_size=8)
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                hovermode='x unified'
            )
            
            if save_path:
                fig.write_html(str(save_path))
                print(f"Interactive chart saved to {save_path}")
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df[date_column], df[stress_column], linewidth=2, marker='o', markersize=6)
            ax.fill_between(df[date_column], df[stress_column], alpha=0.3)
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Stress Level', fontsize=12)
            ax.set_title('Stress Levels Over Time', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {save_path}")
            
            plt.show()
    
    def create_cluster_visualization(self, df, cluster_column='cluster', 
                                    feature_columns=None, save_path=None, interactive=True):
        """
        Create visualization of student clusters
        
        Args:
            df: DataFrame with cluster labels
            cluster_column: Name of cluster column
            feature_columns: Features to visualize (first 2 will be used)
            save_path: Path to save figure
            interactive: Whether to create interactive Plotly chart
        """
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != cluster_column][:2]
        
        if len(feature_columns) < 2:
            raise ValueError("Need at least 2 feature columns for visualization")
        
        if interactive:
            fig = px.scatter(df, x=feature_columns[0], y=feature_columns[1],
                           color=cluster_column,
                           title='Student Clusters',
                           labels={feature_columns[0]: feature_columns[0].replace('_', ' ').title(),
                                  feature_columns[1]: feature_columns[1].replace('_', ' ').title()},
                           hover_data=df.columns.tolist())
            fig.update_traces(marker_size=8, opacity=0.7)
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            
            if save_path:
                fig.write_html(str(save_path))
                print(f"Interactive chart saved to {save_path}")
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(df[feature_columns[0]], df[feature_columns[1]],
                               c=df[cluster_column], cmap='viridis', s=50, alpha=0.6)
            ax.set_xlabel(feature_columns[0].replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(feature_columns[1].replace('_', ' ').title(), fontsize=12)
            ax.set_title('Student Clusters', fontsize=16, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {save_path}")
            
            plt.show()
    
    def create_risk_distribution(self, df, risk_column='risk_group', save_path=None):
        """
        Create distribution visualization of risk groups
        
        Args:
            df: DataFrame with risk classifications
            risk_column: Name of risk group column
            save_path: Path to save figure
        """
        risk_counts = df[risk_column].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['green', 'orange', 'red'])
        axes[0].set_title('Risk Group Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        axes[1].bar(risk_counts.index, risk_counts.values, 
                   color=['green', 'orange', 'red'], alpha=0.7)
        axes[1].set_xlabel('Risk Group', fontsize=12)
        axes[1].set_ylabel('Number of Students', fontsize=12)
        axes[1].set_title('Risk Group Counts', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def create_correlation_heatmap(self, correlation_matrix, save_path=None):
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix: Academic Factors vs Mental Health', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
    
    def create_dashboard_summary(self, df, cluster_stats=None, weekly_stress=None,
                                save_path=None):
        """
        Create comprehensive dashboard summary visualization
        
        Args:
            df: Main DataFrame
            cluster_stats: Cluster statistics DataFrame
            weekly_stress: Weekly stress analysis DataFrame
            save_path: Path to save figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stress Timeline', 'Risk Distribution', 
                          'Cluster Sizes', 'Top Risk Weeks'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Stress timeline
        if weekly_stress is not None:
            period_col = weekly_stress.columns[0]
            fig.add_trace(
                go.Scatter(x=weekly_stress[period_col], 
                          y=weekly_stress['avg_stress'],
                          mode='lines+markers',
                          name='Average Stress'),
                row=1, col=1
            )
        
        # Risk distribution
        if 'risk_group' in df.columns:
            risk_counts = df['risk_group'].value_counts()
            fig.add_trace(
                go.Pie(labels=risk_counts.index, values=risk_counts.values),
                row=1, col=2
            )
        
        # Cluster sizes
        if cluster_stats is not None:
            fig.add_trace(
                go.Bar(x=cluster_stats['cluster_id'], y=cluster_stats['size']),
                row=2, col=1
            )
        
        # Top risk weeks
        if weekly_stress is not None:
            top_5 = weekly_stress.head(5)
            period_col = weekly_stress.columns[0]
            fig.add_trace(
                go.Bar(x=[f"Week {int(w)}" for w in top_5[period_col]], 
                      y=top_5['avg_stress']),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Campus Mental Health Dashboard Summary",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(str(save_path))
            print(f"Dashboard saved to {save_path}")
        
        return fig
    
    def export_for_tableau(self, df, cluster_stats=None, weekly_stress=None,
                          correlation_matrix=None, output_dir='data/outputs/tableau'):
        """
        Export data in formats suitable for Tableau
        
        Args:
            df: Main DataFrame
            cluster_stats: Cluster statistics
            weekly_stress: Weekly stress data
            correlation_matrix: Correlation matrix
            output_dir: Output directory for Tableau files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export main data
        df.to_csv(output_dir / 'student_data.csv', index=False)
        
        if cluster_stats is not None:
            cluster_stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
        
        if weekly_stress is not None:
            weekly_stress.to_csv(output_dir / 'weekly_stress.csv', index=False)
        
        if correlation_matrix is not None:
            correlation_matrix.to_csv(output_dir / 'correlation_matrix.csv')
        
        print(f"Tableau-ready data exported to {output_dir}")


if __name__ == "__main__":
    # Example usage
    visualizer = DashboardVisualizer()
    
    print("Visualization module ready!")
