# Dataset Summary

## Kaggle Student Mental Health Dataset

### Overview
- **Source**: Kaggle (Student Mental Health Dataset)
- **Records**: 101 students
- **Location**: `data/raw/student_mental_health.csv`
- **Date Range**: July-August 2020

### Original Fields
1. **Timestamp** - Survey response date/time
2. **Choose your gender** - Male/Female/Other
3. **Age** - Student age (some missing values)
4. **What is your course?** - Academic course/major
5. **Your current year of Study** - Year 1, 2, 3, or 4
6. **What is your CGPA?** - GPA ranges (0-1.99, 2.00-2.49, 2.50-2.99, 3.00-3.49, 3.50-4.00)
7. **Marital status** - Yes/No
8. **Do you have Depression?** - Yes/No
9. **Do you have Anxiety?** - Yes/No
10. **Do you have Panic attack?** - Yes/No
11. **Did you seek any specialist for a treatment?** - Yes/No

### Adapted Dataset Fields

After running `src/adapt_kaggle_data.py`, the dataset includes:

#### Demographics
- `student_id` - Unique identifier
- `age` - Numeric age (missing values filled)
- `gender` - Gender category
- `course` - Academic course/major
- `year` - Numeric year (1-4)
- `marital_status` - Binary (0/1)

#### Academic Metrics
- `gpa` - Numeric GPA (midpoint of range)
- `course_load` - Estimated course load (12-18 credits)
- `assignment_completion` - Estimated completion % (0-100)
- `exam_scores` - Estimated exam scores (0-100)

#### Mental Health Indicators
- `has_depression` - Binary indicator (0/1)
- `has_anxiety` - Binary indicator (0/1)
- `has_panic_attack` - Binary indicator (0/1)
- `depression_score` - Score 1-10
- `anxiety_score` - Score 1-10
- `panic_score` - Score 1-10
- `stress_score` - Composite score 1-10 (weighted average)
- `mental_health_score` - Same as stress_score
- `total_mental_health_conditions` - Count of conditions (0-3)

#### Lifestyle & Perceptions
- `sleep_hours` - Estimated sleep hours (4-10)
- `workload_perception` - Perceived workload (1-10)

#### Treatment
- `sought_treatment` - Binary (0/1)

#### Temporal
- `date` - Parsed timestamp
- `academic_week` - Academic week number (1-16)

### Key Statistics

After adaptation, you can expect:
- **Average Stress Score**: ~4-6 (depending on prevalence)
- **Depression Prevalence**: ~30-40% (based on Yes responses)
- **Anxiety Prevalence**: ~35-45%
- **Panic Attack Prevalence**: ~25-35%
- **Treatment Seeking**: ~5-10%

### Data Quality Notes

1. **Missing Values**: Some age values are missing (filled with median)
2. **CGPA Ranges**: Converted to numeric using range midpoints
3. **Derived Features**: Some features (course_load, sleep_hours) are estimated based on correlations with available data
4. **Temporal**: Academic weeks are estimated based on timestamp distance from semester start

### Usage

The adapted dataset (`data/processed/kaggle_adapted.csv`) is ready for:
- Clustering analysis
- Correlation analysis
- Statistical modeling
- Temporal analysis
- Visualization

Run the complete analysis:
```bash
python src/run_kaggle_analysis.py
```
