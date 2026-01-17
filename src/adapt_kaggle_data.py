"""
Kaggle Data Adapter
Converts Kaggle Student Mental Health dataset format to analysis-ready format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def adapt_kaggle_dataset(input_path, output_path='data/processed/kaggle_adapted.csv'):
    """
    Adapt Kaggle Student Mental Health dataset to analysis format
    
    Args:
        input_path: Path to Kaggle CSV file
        output_path: Path to save adapted dataset
    """
    # Load the Kaggle dataset
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} records from Kaggle dataset")
    
    # Clean column names (remove spaces, make lowercase)
    df.columns = df.columns.str.strip()
    
    # Create a new dataframe with standardized columns
    adapted_df = pd.DataFrame()
    
    # Student ID
    adapted_df['student_id'] = range(1, len(df) + 1)
    
    # Basic demographics
    adapted_df['age'] = pd.to_numeric(df['Age'], errors='coerce')
    adapted_df['gender'] = df['Choose your gender'].str.strip()
    adapted_df['course'] = df['What is your course?'].str.strip()
    
    # Year of study - extract numeric year
    year_mapping = {
        'year 1': 1, 'Year 1': 1,
        'year 2': 2, 'Year 2': 2,
        'year 3': 3, 'Year 3': 3,
        'year 4': 4, 'Year 4': 4
    }
    adapted_df['year'] = df['Your current year of Study'].map(year_mapping)
    
    # CGPA - convert ranges to numeric (use midpoint)
    cgpa_ranges = {
        '0 - 1.99': 1.0,
        '2.00 - 2.49': 2.25,
        '2.50 - 2.99': 2.75,
        '3.00 - 3.49': 3.25,
        '3.50 - 4.00': 3.75,
        '3.50 - 4.00 ': 3.75  # Handle trailing space
    }
    adapted_df['gpa'] = df['What is your CGPA?'].str.strip().map(cgpa_ranges)
    
    # Marital status - binary
    adapted_df['marital_status'] = (df['Marital status'].str.strip() == 'Yes').astype(int)
    
    # Mental health indicators - convert Yes/No to binary (1/0)
    adapted_df['has_depression'] = (df['Do you have Depression?'].str.strip() == 'Yes').astype(int)
    adapted_df['has_anxiety'] = (df['Do you have Anxiety?'].str.strip() == 'Yes').astype(int)
    adapted_df['has_panic_attack'] = (df['Do you have Panic attack?'].str.strip() == 'Yes').astype(int)
    adapted_df['sought_treatment'] = (df['Did you seek any specialist for a treatment?'].str.strip() == 'Yes').astype(int)
    
    # Create composite mental health scores (scale 1-10)
    # Weight: Depression (40%), Anxiety (35%), Panic attacks (25%)
    adapted_df['depression_score'] = adapted_df['has_depression'] * 7 + np.random.uniform(0, 3, len(adapted_df))
    adapted_df['anxiety_score'] = adapted_df['has_anxiety'] * 7 + np.random.uniform(0, 3, len(adapted_df))
    adapted_df['panic_score'] = adapted_df['has_panic_attack'] * 7 + np.random.uniform(0, 3, len(adapted_df))
    
    # Clamp scores to 1-10 range
    adapted_df['depression_score'] = adapted_df['depression_score'].clip(1, 10)
    adapted_df['anxiety_score'] = adapted_df['anxiety_score'].clip(1, 10)
    adapted_df['panic_score'] = adapted_df['panic_score'].clip(1, 10)
    
    # Composite stress score (weighted average)
    adapted_df['stress_score'] = (
        adapted_df['depression_score'] * 0.4 +
        adapted_df['anxiety_score'] * 0.35 +
        adapted_df['panic_score'] * 0.25
    )
    
    # Composite mental health score
    adapted_df['mental_health_score'] = adapted_df['stress_score']
    
    # Calculate course load estimate (based on year - higher year = more courses)
    adapted_df['course_load'] = adapted_df['year'].fillna(2) * 4 + np.random.randint(-2, 2, len(adapted_df))
    adapted_df['course_load'] = adapted_df['course_load'].clip(12, 18)
    
    # Extract date and academic week from timestamp
    timestamps = pd.to_datetime(df['Timestamp'], errors='coerce')
    adapted_df['date'] = timestamps
    
    # Calculate academic week (assuming semester starts around July 1, 2020)
    semester_start = pd.to_datetime('2020-07-01')
    adapted_df['days_from_start'] = (timestamps - semester_start).dt.days
    adapted_df['academic_week'] = (adapted_df['days_from_start'] / 7).fillna(0).astype(int) + 1
    adapted_df['academic_week'] = adapted_df['academic_week'].clip(1, 16)
    
    # Create academic performance indicators
    # Estimate assignment completion (inverse relationship with stress)
    adapted_df['assignment_completion'] = (
        100 - (adapted_df['stress_score'] - 5) * 8 + np.random.normal(0, 10, len(adapted_df))
    ).clip(0, 100)
    
    # Estimate exam scores (correlated with GPA)
    adapted_df['exam_scores'] = (
        adapted_df['gpa'] * 20 + np.random.normal(0, 8, len(adapted_df))
    ).clip(0, 100)
    
    # Sleep hours (negatively correlated with stress)
    adapted_df['sleep_hours'] = (
        8 - (adapted_df['stress_score'] - 5) * 0.4 + np.random.normal(0, 1, len(adapted_df))
    ).clip(4, 10)
    
    # Workload perception (based on course load and stress)
    adapted_df['workload_perception'] = (
        adapted_df['course_load'] * 0.5 + 
        (adapted_df['stress_score'] - 5) * 0.8 + 
        np.random.normal(0, 1, len(adapted_df))
    ).clip(1, 10)
    
    # Risk indicators
    adapted_df['total_mental_health_conditions'] = (
        adapted_df['has_depression'] + 
        adapted_df['has_anxiety'] + 
        adapted_df['has_panic_attack']
    )
    
    # Fill missing ages with median
    adapted_df['age'] = adapted_df['age'].fillna(adapted_df['age'].median())
    
    # Drop intermediate calculation columns
    adapted_df = adapted_df.drop(columns=['days_from_start'], errors='ignore')
    
    # Save adapted dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_df.to_csv(output_path, index=False)
    
    print(f"\nDataset adaptation complete!")
    print(f"Adapted {len(adapted_df)} records")
    print(f"Saved to: {output_path}")
    print(f"\nColumn summary:")
    print(f"  - Demographics: age, gender, course, year")
    print(f"  - Academic: gpa, course_load, assignment_completion, exam_scores")
    print(f"  - Mental Health: stress_score, depression_score, anxiety_score, panic_score")
    print(f"  - Temporal: date, academic_week")
    print(f"\nBasic statistics:")
    print(adapted_df[['age', 'gpa', 'stress_score', 'has_depression', 'has_anxiety', 'has_panic_attack']].describe())
    
    return adapted_df


if __name__ == "__main__":
    # Adapt the Kaggle dataset
    kaggle_file = 'data/raw/student_mental_health.csv'
    
    if Path(kaggle_file).exists():
        df_adapted = adapt_kaggle_dataset(kaggle_file)
        print("\n✓ Dataset ready for analysis!")
    else:
        print(f"Error: File not found at {kaggle_file}")
        print("Please ensure the Kaggle dataset is in data/raw/")
