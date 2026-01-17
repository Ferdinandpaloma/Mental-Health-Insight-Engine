# Using the Kaggle Student Mental Health Dataset

## Dataset Information

The Kaggle dataset (`student_mental_health.csv`) contains 101 student records with the following fields:
- **Demographics**: Gender, Age, Course, Year of Study
- **Academic**: CGPA (in ranges)
- **Mental Health Indicators**: Depression (Yes/No), Anxiety (Yes/No), Panic Attacks (Yes/No)
- **Treatment**: Whether student sought specialist treatment
- **Temporal**: Timestamp of survey response

## Quick Start with Kaggle Data

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Adapt the Kaggle Dataset

The Kaggle dataset format needs to be adapted to work with our analysis pipeline:

```bash
python src/adapt_kaggle_data.py
```

This will:
- Convert Yes/No indicators to binary (1/0)
- Convert CGPA ranges to numeric values
- Extract academic week from timestamps
- Create composite stress and mental health scores
- Generate additional derived features (course load, sleep hours, etc.)

### Step 3: Run the Analysis Pipeline

```bash
python src/main.py data/processed/kaggle_adapted.csv
```

Or using the adapted script specifically:

```bash
python src/run_kaggle_analysis.py
```

## Dataset Adaptations Made

### Column Mappings

| Kaggle Column | Adapted Column | Transformation |
|--------------|----------------|----------------|
| `Age` | `age` | Numeric, missing values filled with median |
| `Choose your gender` | `gender` | String (cleaned) |
| `What is your course?` | `course` | String (cleaned) |
| `Your current year of Study` | `year` | Numeric (1-4) |
| `What is your CGPA?` | `gpa` | Numeric (midpoint of range) |
| `Marital status` | `marital_status` | Binary (0/1) |
| `Do you have Depression?` | `has_depression`, `depression_score` | Binary + Score (1-10) |
| `Do you have Anxiety?` | `has_anxiety`, `anxiety_score` | Binary + Score (1-10) |
| `Do you have Panic attack?` | `has_panic_attack`, `panic_score` | Binary + Score (1-10) |
| `Timestamp` | `date`, `academic_week` | Date + Week number |

### Derived Features Created

1. **Stress Score**: Weighted composite of depression, anxiety, and panic scores
2. **Mental Health Score**: Same as stress score for consistency
3. **Course Load**: Estimated based on year of study (12-18 credits)
4. **Assignment Completion**: Derived from stress levels (inverse relationship)
5. **Exam Scores**: Derived from GPA with some variance
6. **Sleep Hours**: Derived from stress levels (inverse relationship)
7. **Workload Perception**: Combination of course load and stress

## Analysis Output

After running the analysis, you'll get:

1. **Cluster Analysis**: Identifies distinct student groups based on mental health patterns
2. **Correlation Analysis**: Relationships between academic factors and mental health
3. **Peak Stress Periods**: Temporal patterns in stress levels
4. **At-Risk Classification**: Statistical models classifying students by risk level
5. **Visualizations**: Charts and graphs for dashboard creation
6. **Tableau Exports**: Ready-to-use data files for Tableau visualization

## Key Findings from Kaggle Dataset

The adapted dataset enables analysis of:
- Prevalence of depression, anxiety, and panic attacks
- Relationships between academic performance (GPA) and mental health
- Year-of-study effects on mental health
- Course/field differences in mental health indicators
- Treatment-seeking behavior patterns

## Notes

- The Kaggle dataset has some missing values (especially in Age field)
- CGPA is provided in ranges, so numeric values are midpoints
- Some derived features (course load, sleep hours) are estimated based on correlations
- The dataset is smaller (101 records) so some statistical tests may have limited power

## Customization

You can modify `src/adapt_kaggle_data.py` to:
- Adjust scoring weights for mental health indicators
- Change how academic week is calculated
- Modify derived feature calculations
- Add additional transformations
