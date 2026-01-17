"""
Student Habits & Performance Data Adapter
Converts habits/performance dataset to analysis-ready format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def adapt_habits_dataset(input_path, output_path='data/processed/habits_adapted.csv'):
    """
    Adapt Student Habits & Performance dataset to analysis format
    
    Args:
        input_path: Path to habits CSV file
        output_path: Path to save adapted dataset
    """
    # Load the dataset
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} records from Habits & Performance dataset")
    
    # Create adapted dataframe
    adapted_df = pd.DataFrame()
    
    # Student ID
    adapted_df['student_id'] = df['student_id'].str.replace('S', '').astype(int)
    
    # Demographics
    adapted_df['age'] = df['age'].astype(float)
    adapted_df['gender'] = df['gender'].str.strip()
    
    # Academic metrics
    adapted_df['study_hours_per_day'] = df['study_hours_per_day'].astype(float)
    adapted_df['attendance_percentage'] = df['attendance_percentage'].astype(float)
    adapted_df['exam_score'] = df['exam_score'].astype(float)
    
    # Calculate GPA from exam scores (estimate)
    adapted_df['gpa'] = (df['exam_score'] / 25).clip(0, 4.0)
    
    # Course load estimate (based on study hours - assume 15 credit hours for 5 hours study/day)
    adapted_df['course_load'] = ((df['study_hours_per_day'] / 5) * 15).clip(12, 18).round()
    
    # Lifestyle factors
    adapted_df['sleep_hours'] = df['sleep_hours'].astype(float)
    adapted_df['social_media_hours'] = df['social_media_hours'].astype(float)
    adapted_df['netflix_hours'] = df['netflix_hours'].astype(float)
    adapted_df['exercise_frequency'] = df['exercise_frequency'].astype(float)
    
    # Diet quality - convert to numeric (Poor=1, Fair=2, Good=3)
    diet_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3}
    adapted_df['diet_quality_score'] = df['diet_quality'].map(diet_mapping)
    
    # Part-time job - binary
    adapted_df['part_time_job'] = (df['part_time_job'].str.strip() == 'Yes').astype(int)
    
    # Extracurricular participation - binary
    adapted_df['extracurricular_participation'] = (df['extracurricular_participation'].str.strip() == 'Yes').astype(int)
    
    # Parental education - convert to numeric (None=0, High School=1, Bachelor=2, Master=3)
    parental_edu_mapping = {
        'None': 0,
        'High School': 1,
        'Bachelor': 2,
        'Master': 3
    }
    adapted_df['parental_education_level'] = df['parental_education_level'].map(parental_edu_mapping)
    
    # Internet quality - convert to numeric (Poor=1, Average=2, Good=3)
    internet_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    adapted_df['internet_quality_score'] = df['internet_quality'].map(internet_mapping)
    
    # Mental health rating (already numeric 1-10)
    adapted_df['mental_health_rating'] = df['mental_health_rating'].astype(float)
    
    # Create composite stress score from mental health rating and lifestyle factors
    # Higher screen time, lower sleep, lower exercise = higher stress
    adapted_df['stress_score'] = (
        (10 - adapted_df['mental_health_rating']) * 0.5 +  # Mental health (inverted)
        ((adapted_df['social_media_hours'] + adapted_df['netflix_hours']) / 10) * 2 +  # Screen time
        (8 - adapted_df['sleep_hours']) * 0.3 +  # Sleep deficit
        (7 - adapted_df['exercise_frequency']) * 0.2  # Exercise deficit
    ).clip(1, 10)
    
    # Create anxiety and depression scores based on stress score
    adapted_df['anxiety_score'] = (adapted_df['stress_score'] * 0.85 + np.random.uniform(0, 1.5, len(adapted_df))).clip(1, 10)
    adapted_df['depression_score'] = (adapted_df['stress_score'] * 0.75 + np.random.uniform(0, 2, len(adapted_df))).clip(1, 10)
    
    # Binary mental health indicators
    adapted_df['has_depression'] = (adapted_df['depression_score'] >= 7).astype(int)
    adapted_df['has_anxiety'] = (adapted_df['anxiety_score'] >= 7).astype(int)
    adapted_df['has_panic_attack'] = ((adapted_df['stress_score'] >= 7) & (adapted_df['anxiety_score'] >= 8)).astype(int)
    
    # Composite mental health score
    adapted_df['mental_health_score'] = adapted_df['mental_health_rating']
    
    # Assignment completion (estimate from attendance and study hours)
    adapted_df['assignment_completion'] = (
        adapted_df['attendance_percentage'] * 0.7 + 
        (adapted_df['study_hours_per_day'] / 8 * 100) * 0.3
    ).clip(0, 100)
    
    # Create workload perception
    adapted_df['workload_perception'] = (
        (adapted_df['study_hours_per_day'] - 3) * 1.5 + 
        (adapted_df['stress_score'] - 5) * 0.8
    ).clip(1, 10)
    
    # Total screen time
    adapted_df['total_screen_time'] = adapted_df['social_media_hours'] + adapted_df['netflix_hours']
    
    # Total mental health conditions
    adapted_df['total_mental_health_conditions'] = (
        adapted_df['has_depression'] + 
        adapted_df['has_anxiety'] + 
        adapted_df['has_panic_attack']
    )
    
    # Generate academic week (random distribution for temporal analysis)
    np.random.seed(42)
    adapted_df['academic_week'] = np.random.randint(1, 17, len(adapted_df))
    adapted_df['date'] = datetime(2023, 9, 1) + pd.to_timedelta((adapted_df['academic_week'] - 1) * 7, unit='d')
    
    # Year of study estimate (based on age - assume 18 = year 1)
    adapted_df['year'] = ((adapted_df['age'] - 18) + 1).clip(1, 4).astype(int)
    
    # Save adapted dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_df.to_csv(output_path, index=False)
    
    print(f"\nDataset adaptation complete!")
    print(f"Adapted {len(adapted_df)} records")
    print(f"Saved to: {output_path}")
    print(f"\nColumn summary:")
    print(f"  - Demographics: age, gender, year")
    print(f"  - Academic: gpa, exam_score, attendance_percentage, study_hours_per_day")
    print(f"  - Lifestyle: sleep_hours, diet_quality_score, exercise_frequency, total_screen_time")
    print(f"  - Mental Health: stress_score, anxiety_score, depression_score, mental_health_rating")
    print(f"  - Temporal: date, academic_week")
    print(f"\nBasic statistics:")
    print(adapted_df[['age', 'gpa', 'stress_score', 'mental_health_rating', 'exam_score']].describe())
    
    return adapted_df


if __name__ == "__main__":
    # Adapt the habits dataset
    habits_file = 'data/raw/student_habits_performance.csv'
    
    if Path(habits_file).exists():
        df_adapted = adapt_habits_dataset(habits_file)
        print("\n✓ Dataset ready for analysis!")
    else:
        print(f"Error: File not found at {habits_file}")
        print("Please ensure the habits dataset is in data/raw/")
