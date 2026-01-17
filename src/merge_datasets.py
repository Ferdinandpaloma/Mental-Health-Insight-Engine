"""
Dataset Merger
Merges Kaggle mental health dataset with Habits & Performance dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def merge_datasets(kaggle_path='data/processed/kaggle_adapted.csv',
                  habits_path='data/processed/habits_adapted.csv',
                  output_path='data/processed/combined_dataset.csv'):
    """
    Merge Kaggle and Habits datasets for comprehensive analysis
    
    Args:
        kaggle_path: Path to adapted Kaggle dataset
        habits_path: Path to adapted habits dataset
        output_path: Path to save merged dataset
    """
    # Load both datasets
    print("Loading datasets...")
    df_kaggle = pd.read_csv(kaggle_path) if Path(kaggle_path).exists() else None
    df_habits = pd.read_csv(habits_path) if Path(habits_path).exists() else None
    
    if df_kaggle is None and df_habits is None:
        raise ValueError("No datasets found. Please run adapters first.")
    
    datasets = []
    
    # Add Kaggle dataset with source marker
    if df_kaggle is not None:
        df_kaggle['data_source'] = 'kaggle'
        # Rename student_id to ensure uniqueness
        df_kaggle['student_id'] = range(1, len(df_kaggle) + 1)
        datasets.append(df_kaggle)
        print(f"  - Kaggle dataset: {len(df_kaggle)} records")
    
    # Add Habits dataset with source marker
    if df_habits is not None:
        df_habits['data_source'] = 'habits'
        # Adjust student_id to avoid conflicts with Kaggle
        if df_kaggle is not None:
            df_habits['student_id'] = df_habits['student_id'] + len(df_kaggle) + 10000
        datasets.append(df_habits)
        print(f"  - Habits dataset: {len(df_habits)} records")
    
    # Find common columns
    if len(datasets) == 2:
        common_cols = set(datasets[0].columns) & set(datasets[1].columns)
        print(f"\nCommon columns: {len(common_cols)}")
        
        # Merge datasets
        df_combined = pd.concat(datasets, ignore_index=True, sort=False)
    else:
        df_combined = datasets[0]
    
    # Ensure all numeric columns are filled
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_combined[col].isna().any():
            df_combined[col].fillna(df_combined[col].median(), inplace=True)
    
    # Save merged dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Datasets merged successfully!")
    print(f"  Total records: {len(df_combined)}")
    print(f"  Total columns: {len(df_combined.columns)}")
    print(f"  Saved to: {output_path}")
    
    # Show data source distribution
    if 'data_source' in df_combined.columns:
        print(f"\nData source distribution:")
        print(df_combined['data_source'].value_counts())
    
    # Show key statistics
    print(f"\nKey statistics for merged dataset:")
    key_cols = ['age', 'gpa', 'stress_score', 'exam_score', 'sleep_hours']
    available_cols = [col for col in key_cols if col in df_combined.columns]
    if available_cols:
        print(df_combined[available_cols].describe())
    
    return df_combined


if __name__ == "__main__":
    # Merge datasets
    combined_df = merge_datasets()
    print("\n✓ Merged dataset ready for analysis!")
