"""
Sample Data Generator
Generates synthetic student mental health data for testing and demonstration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_sample_data(n_students=10000, output_path='data/raw/student_data.csv'):
    """
    Generate synthetic student mental health data
    
    Args:
        n_students: Number of student records to generate
        output_path: Path to save the generated data
    """
    np.random.seed(42)
    
    # Generate base student data
    data = {
        'student_id': range(1, n_students + 1),
        'age': np.random.randint(18, 25, n_students),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_students, p=[0.45, 0.50, 0.05]),
        'year': np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], n_students),
        'gpa': np.random.normal(3.2, 0.6, n_students).clip(0, 4.0),
        'course_load': np.random.randint(12, 18, n_students),
    }
    
    # Generate stress-related variables (correlated with academic factors)
    # Stress increases with course load and decreases with GPA
    base_stress = 5.0
    stress_score = base_stress + (data['course_load'] - 15) * 0.3 - (data['gpa'] - 3.2) * 0.8
    stress_score += np.random.normal(0, 1.5, n_students)
    data['stress_score'] = stress_score.clip(1, 10)
    
    # Generate anxiety and depression scores (correlated with stress)
    data['anxiety_score'] = (data['stress_score'] * 0.7 + np.random.normal(0, 1, n_students)).clip(1, 10)
    data['depression_score'] = (data['stress_score'] * 0.6 + np.random.normal(0, 1.2, n_students)).clip(1, 10)
    
    # Generate sleep hours (negatively correlated with stress)
    data['sleep_hours'] = (8 - (data['stress_score'] - 5) * 0.3 + np.random.normal(0, 1, n_students)).clip(4, 10)
    
    # Generate academic performance metrics
    data['assignment_completion'] = (100 - (data['stress_score'] - 5) * 5 + np.random.normal(0, 10, n_students)).clip(0, 100)
    data['exam_scores'] = (data['gpa'] * 20 + np.random.normal(0, 8, n_students)).clip(0, 100)
    
    # Generate temporal data (academic weeks)
    start_date = datetime(2023, 9, 1)  # Start of fall semester
    weeks = []
    dates = []
    
    for i in range(n_students):
        # Random week in semester (1-16 weeks)
        week = np.random.randint(1, 17)
        weeks.append(week)
        dates.append(start_date + timedelta(weeks=week-1))
    
    data['academic_week'] = weeks
    data['date'] = dates
    
    # Create composite mental health score
    data['mental_health_score'] = (
        data['stress_score'] * 0.4 + 
        data['anxiety_score'] * 0.3 + 
        data['depression_score'] * 0.3
    )
    
    # Generate workload perception
    data['workload_perception'] = (
        data['course_load'] * 0.5 + 
        (data['assignment_completion'] / 10) * 0.3 + 
        np.random.normal(0, 1, n_students)
    ).clip(1, 10)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some noise and missing values (realistic scenario)
    # Randomly set 2% of stress scores to missing
    missing_indices = np.random.choice(df.index, size=int(n_students * 0.02), replace=False)
    df.loc[missing_indices, 'stress_score'] = np.nan
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_students:,} student records")
    print(f"Data saved to: {output_path}")
    print(f"\nData Summary:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(n_students=10000)
    print("\nSample data generation complete!")
    print("You can now run the analysis pipeline with this data.")
