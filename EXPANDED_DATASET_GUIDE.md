# Expanded Dataset Guide

## Overview

The project now supports **two complementary datasets** that provide a comprehensive view of student mental health:

1. **Kaggle Student Mental Health Dataset** (101 records)
   - Mental health conditions (Depression, Anxiety, Panic Attacks)
   - Treatment-seeking behavior
   - Academic demographics (CGPA, Year of Study)

2. **Student Habits & Performance Dataset** (1,000 records)
   - Lifestyle factors (sleep, diet, exercise)
   - Digital habits (social media, Netflix)
   - Academic performance (exam scores, attendance, study hours)
   - Mental health ratings

**Combined Total: ~1,100 student records**

## Dataset Features

### Combined Dataset Includes:

#### Demographics
- `age`, `gender`, `year` (year of study)

#### Academic Metrics
- `gpa` - Grade Point Average
- `exam_score` - Exam performance scores
- `attendance_percentage` - Class attendance
- `study_hours_per_day` - Daily study hours
- `course_load` - Estimated course load
- `assignment_completion` - Assignment completion rate

#### Mental Health Indicators
- `stress_score` - Composite stress score (1-10)
- `depression_score` - Depression score (1-10)
- `anxiety_score` - Anxiety score (1-10)
- `mental_health_rating` - Overall mental health rating (1-10)
- `has_depression` - Binary indicator
- `has_anxiety` - Binary indicator
- `has_panic_attack` - Binary indicator
- `total_mental_health_conditions` - Count of conditions

#### Lifestyle Factors
- `sleep_hours` - Daily sleep hours
- `diet_quality_score` - Diet quality (1-3: Poor/Fair/Good)
- `exercise_frequency` - Exercise frequency
- `total_screen_time` - Combined social media + Netflix hours
- `social_media_hours` - Daily social media usage
- `netflix_hours` - Daily Netflix/streaming hours

#### Social & Economic
- `part_time_job` - Whether student works part-time
- `extracurricular_participation` - Extracurricular activities
- `parental_education_level` - Parent education (0-3)
- `internet_quality_score` - Internet quality (1-3)

#### Temporal
- `date` - Record date
- `academic_week` - Academic week number (1-16)

#### Metadata
- `student_id` - Unique student identifier
- `data_source` - Dataset source ('kaggle' or 'habits')

## Usage

### Quick Start with Expanded Analysis

Run the expanded analysis pipeline that processes both datasets:

```bash
python src/run_expanded_analysis.py
```

This will:
1. Adapt the Kaggle dataset (if available)
2. Adapt the Habits & Performance dataset
3. Merge both datasets
4. Run complete analysis on combined data

### Individual Dataset Processing

#### Process only Habits dataset:

```bash
python src/adapt_habits_data.py
python src/main.py data/processed/habits_adapted.csv
```

#### Process only Kaggle dataset:

```bash
python src/adapt_kaggle_data.py
python src/main.py data/processed/kaggle_adapted.csv
```

#### Merge existing adapted datasets:

```bash
python src/merge_datasets.py
python src/main.py data/processed/combined_dataset.csv
```

## Analysis Benefits

The expanded dataset enables analysis of:

### 1. Lifestyle-Mental Health Correlations
- **Sleep patterns** vs stress levels
- **Screen time** impact on mental health
- **Exercise frequency** and mental wellness
- **Diet quality** and stress management

### 2. Academic Performance Relationships
- Study hours vs exam scores
- Attendance vs mental health
- Workload perception vs stress

### 3. Digital Habits Analysis
- Social media usage patterns
- Streaming habits
- Screen time effects on well-being

### 4. Socioeconomic Factors
- Parental education impact
- Part-time employment effects
- Internet quality influence

### 5. Comprehensive Clustering
- More robust cluster identification (1,100+ records vs 101)
- Better statistical power for modeling
- More reliable correlation patterns

## Data Adaptation Details

### Habits Dataset Adaptations

- **Mental Health Scores**: Derived from `mental_health_rating` and lifestyle factors
- **Stress Score**: Composite of mental health rating, screen time, sleep, and exercise
- **Academic Metrics**: GPA estimated from exam scores, course load from study hours
- **Binary Indicators**: Created from continuous scores using thresholds

### Dataset Merging

- **Unique IDs**: Student IDs adjusted to avoid conflicts
- **Common Columns**: Aligned across both datasets
- **Missing Values**: Filled with medians for numeric columns
- **Source Tracking**: `data_source` column indicates origin

## Output Files

After running expanded analysis:

### Processed Data
- `data/processed/habits_adapted.csv` - Adapted habits dataset
- `data/processed/combined_dataset.csv` - Merged dataset (if both sources available)

### Analysis Results
- `data/outputs/cluster_statistics.csv` - Enhanced clustering with more records
- `data/outputs/correlation_report.xlsx` - Expanded correlations including lifestyle factors
- `data/outputs/model_evaluation.csv` - More robust model performance metrics
- `data/outputs/student_classifications.csv` - All student risk classifications

### Visualizations
- Enhanced cluster visualizations with more data points
- Lifestyle factor correlations
- Screen time vs mental health charts
- Academic performance relationships

## Key Insights from Expanded Dataset

With 1,100+ records, you can now:

1. **Identify Lifestyle Patterns**: Screen time, sleep, and exercise correlations
2. **Academic-Stress Relationships**: Study hours, attendance, and performance impacts
3. **Digital Wellness**: Social media and streaming effects on mental health
4. **Robust Clustering**: More reliable identification of at-risk student groups
5. **Better Predictions**: Improved statistical modeling with larger sample size

## Next Steps

1. Run the expanded analysis: `python src/run_expanded_analysis.py`
2. Explore lifestyle correlations in correlation report
3. Review enhanced cluster visualizations
4. Analyze screen time and sleep patterns in Tableau
5. Build predictive models with lifestyle factors

## Notes

- The habits dataset provides more lifestyle detail than the Kaggle dataset
- Combined analysis offers comprehensive insights
- Both datasets complement each other for holistic understanding
- Analysis pipeline works with either dataset individually or combined
