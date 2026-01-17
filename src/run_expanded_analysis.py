"""
Expanded Analysis Pipeline
Runs analysis on combined Kaggle + Habits datasets
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapt_kaggle_data import adapt_kaggle_dataset
from src.adapt_habits_data import adapt_habits_dataset
from src.merge_datasets import merge_datasets
from src.main import main_analysis_pipeline


def run_expanded_analysis():
    """Run complete analysis pipeline on combined datasets"""
    
    print("=" * 70)
    print("Expanded Campus Mental Health Analysis - Combined Datasets")
    print("=" * 70)
    
    # Step 1: Adapt Kaggle dataset
    print("\n[Step 1/4] Adapting Kaggle Mental Health dataset...")
    print("-" * 70)
    
    kaggle_input = 'data/raw/student_mental_health.csv'
    kaggle_output = 'data/processed/kaggle_adapted.csv'
    
    if Path(kaggle_input).exists():
        try:
            df_kaggle = adapt_kaggle_dataset(kaggle_input, kaggle_output)
            print("✓ Kaggle dataset adapted!")
        except Exception as e:
            print(f"Warning: Could not adapt Kaggle dataset: {e}")
            kaggle_output = None
    else:
        print(f"Warning: Kaggle dataset not found at {kaggle_input}")
        kaggle_output = None
    
    # Step 2: Adapt Habits dataset
    print("\n[Step 2/4] Adapting Student Habits & Performance dataset...")
    print("-" * 70)
    
    habits_input = 'data/raw/student_habits_performance.csv'
    habits_output = 'data/processed/habits_adapted.csv'
    
    if Path(habits_input).exists():
        try:
            df_habits = adapt_habits_dataset(habits_input, habits_output)
            print("✓ Habits dataset adapted!")
        except Exception as e:
            print(f"Error: Could not adapt habits dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Error: Habits dataset not found at {habits_input}")
        return None
    
    # Step 3: Merge datasets
    print("\n[Step 3/4] Merging datasets...")
    print("-" * 70)
    
    try:
        combined_df = merge_datasets(
            kaggle_path=kaggle_output if kaggle_output else None,
            habits_path=habits_output,
            output_path='data/processed/combined_dataset.csv'
        )
        print("✓ Datasets merged!")
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        # If merge fails, use just habits dataset
        combined_path = habits_output
        print(f"Using habits dataset only: {combined_path}")
    else:
        combined_path = 'data/processed/combined_dataset.csv'
    
    # Step 4: Run analysis pipeline
    print("\n[Step 4/4] Running analysis pipeline on combined dataset...")
    print("-" * 70)
    
    try:
        results = main_analysis_pipeline(combined_path)
        print("\n✓ Analysis pipeline complete!")
        return results
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_expanded_analysis()
    
    if results:
        print("\n" + "=" * 70)
        print("EXPANDED ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nAll outputs saved to: data/outputs/")
        print("\nDataset includes:")
        print("  - Kaggle Mental Health data (101 records)")
        print("  - Student Habits & Performance data (1,000 records)")
        print("  - Combined total: ~1,100+ records")
        print("\nNext steps:")
        print("1. Review visualizations in data/outputs/")
        print("2. Import Tableau-ready data from data/outputs/tableau/")
        print("3. Explore the enhanced analysis with lifestyle factors")
