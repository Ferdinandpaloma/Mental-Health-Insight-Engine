"""
Main Analysis Pipeline
Orchestrates the complete analysis workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.clustering import StudentClustering
from src.correlation_analysis import CorrelationAnalyzer
from src.statistical_modeling import AtRiskModeling
from src.visualization import DashboardVisualizer

import warnings
warnings.filterwarnings('ignore')


def main_analysis_pipeline(data_path, output_dir='data/outputs'):
    """
    Complete analysis pipeline for Campus Mental Health Insight Engine
    
    Args:
        data_path: Path to raw data file
        output_dir: Directory to save all outputs
    """
    print("=" * 60)
    print("Campus Mental Health Insight Engine - Analysis Pipeline")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n[1/5] Data Preprocessing...")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(
        file_path=data_path,
        save_path=output_path / 'processed_data.csv'
    )
    print(f"✓ Processed {len(df_processed)} student records")
    
    # Step 2: Clustering Analysis
    print("\n[2/5] Clustering Analysis...")
    print("-" * 60)
    cluster_analyzer = StudentClustering(n_clusters=3)
    df_clustered, cluster_stats = cluster_analyzer.cluster_pipeline(
        df_processed,
        method='kmeans',
        visualize=True
    )
    cluster_stats.to_csv(output_path / 'cluster_statistics.csv', index=False)
    print("✓ Identified distinct student groups")
    
    # Step 3: Correlation Analysis
    print("\n[3/5] Correlation Analysis...")
    print("-" * 60)
    corr_analyzer = CorrelationAnalyzer()
    correlation_report = corr_analyzer.generate_correlation_report(
        df_processed,
        save_path=output_path / 'correlation_report.xlsx'
    )
    
    # Detect peak stress periods
    weekly_stress = corr_analyzer.detect_peak_stress_periods(df_processed)
    weekly_stress.to_csv(output_path / 'weekly_stress_analysis.csv', index=False)
    
    print("✓ Identified stress patterns and peak periods")
    print(f"✓ Top 5 high-risk weeks identified")
    
    # Step 4: Statistical Modeling
    print("\n[4/5] Statistical Modeling...")
    print("-" * 60)
    modeler = AtRiskModeling()
    modeling_results = modeler.modeling_pipeline(
        df_clustered,
        save_results=True
    )
    print("✓ Built predictive models for at-risk classification")
    
    # Step 5: Visualization and Dashboard Preparation
    print("\n[5/5] Creating Visualizations...")
    print("-" * 60)
    visualizer = DashboardVisualizer()
    
    # Create key visualizations
    if 'academic_week' in df_processed.columns or 'week' in df_processed.columns:
        date_col = 'academic_week' if 'academic_week' in df_processed.columns else 'week'
        stress_col = [col for col in df_processed.columns if 'stress' in col.lower()]
        if stress_col:
            visualizer.create_stress_timeline(
                weekly_stress,
                date_column=weekly_stress.columns[0],
                stress_column='avg_stress',
                save_path=output_path / 'stress_timeline.html',
                interactive=True
            )
    
    # Risk distribution
    if 'risk_group' in df_clustered.columns:
        visualizer.create_risk_distribution(
            df_clustered,
            save_path=output_path / 'risk_distribution.png'
        )
    
    # Export for Tableau
    visualizer.export_for_tableau(
        df_clustered,
        cluster_stats=cluster_stats,
        weekly_stress=weekly_stress,
        correlation_matrix=correlation_report['correlation_matrix'],
        output_dir=output_path / 'tableau'
    )
    
    print("✓ Visualizations created")
    print("✓ Tableau-ready data exported")
    
    # Summary Report
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Students Analyzed: {len(df_processed):,}")
    print(f"Distinct Student Groups Identified: {len(cluster_stats)}")
    print(f"High-Risk Academic Weeks: 5")
    print(f"At-Risk Students Identified: {df_clustered['at_risk'].sum() if 'at_risk' in df_clustered.columns else 'N/A'}")
    print(f"\nAll outputs saved to: {output_path}")
    print("=" * 60)
    
    return {
        'processed_data': df_processed,
        'clustered_data': df_clustered,
        'cluster_stats': cluster_stats,
        'correlation_report': correlation_report,
        'weekly_stress': weekly_stress,
        'modeling_results': modeling_results
    }


if __name__ == "__main__":
    # Example usage
    # Update with your actual data path
    data_file = 'data/raw/student_data.csv'
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    if not Path(data_file).exists():
        print(f"Warning: Data file not found at {data_file}")
        print("Please provide a valid data file path.")
        print("\nUsage: python main.py <path_to_data_file>")
        print("Example: python main.py data/raw/student_data.csv")
    else:
        results = main_analysis_pipeline(data_file)
