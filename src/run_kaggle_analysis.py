"""
Complete Analysis Pipeline for Kaggle Dataset
Runs data adaptation and full analysis in one script
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapt_kaggle_data import adapt_kaggle_dataset
from src.main import main_analysis_pipeline


def run_kaggle_analysis():
    """Run complete analysis pipeline on Kaggle dataset"""
    
    print("=" * 70)
    print("Kaggle Student Mental Health Dataset - Complete Analysis")
    print("=" * 70)
    
    # Step 1: Adapt the dataset
    print("\n[Step 1/2] Adapting Kaggle dataset format...")
    print("-" * 70)
    
    kaggle_input = 'data/raw/student_mental_health.csv'
    adapted_output = 'data/processed/kaggle_adapted.csv'
    
    if not Path(kaggle_input).exists():
        print(f"Error: Kaggle dataset not found at {kaggle_input}")
        print("Please ensure the file exists in data/raw/")
        return None
    
    try:
        df_adapted = adapt_kaggle_dataset(kaggle_input, adapted_output)
        print("\n✓ Dataset adaptation complete!")
    except Exception as e:
        print(f"Error during adaptation: {e}")
        return None
    
    # Step 2: Run the analysis pipeline
    print("\n[Step 2/2] Running analysis pipeline...")
    print("-" * 70)
    
    try:
        results = main_analysis_pipeline(adapted_output)
        print("\n✓ Analysis pipeline complete!")
        return results
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_kaggle_analysis()
    
    if results:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nAll outputs saved to: data/outputs/")
        print("\nNext steps:")
        print("1. Review visualizations in data/outputs/")
        print("2. Import Tableau-ready data from data/outputs/tableau/")
        print("3. Explore the Jupyter notebook: notebooks/exploratory_analysis.ipynb")
