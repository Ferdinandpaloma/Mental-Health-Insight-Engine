# Campus Mental Health Insight Engine

**December 2023 – May 2024**

A comprehensive data analytics project that analyzes student mental health patterns using advanced clustering, correlation analysis, and statistical modeling to identify at-risk student groups and peak stress periods.

## Project Overview

This project analyzes 10,000+ simulated student data points to:
- Detect campus stress patterns using clustering and correlation analysis
- Identify peak stress periods and model high-risk academic weeks
- Classify distinct at-risk student groups through statistical modeling
- Provide actionable insights for proactive wellness interventions

## Key Features

### 1. Stress Pattern Detection
- **Clustering Analysis**: Identifies distinct student groups based on stress indicators
- **Correlation Analysis**: Discovers relationships between academic factors and mental health metrics
- **Temporal Analysis**: Tracks stress patterns across academic calendar

### 2. Interactive Dashboard
- Visualizes peak stress periods throughout the academic year
- Models 5 high-risk academic weeks for proactive intervention
- Provides real-time insights for campus wellness programs

### 3. At-Risk Student Identification
- **Statistical Modeling**: Uses advanced algorithms to classify student risk levels
- **3 Distinct Groups**: Identifies unique at-risk student profiles
- **Predictive Analytics**: Enables proactive outreach and support

## Project Structure

```
Mental-health/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and processed data
│   └── outputs/          # Analysis results and visualizations
├── src/
│   ├── data_preprocessing.py
│   ├── clustering.py
│   ├── correlation_analysis.py
│   ├── statistical_modeling.py
│   └── visualization.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── dashboard_creation.ipynb
└── docs/
    └── methodology.md
```

## Technologies Used

- **Python**: Data analysis and modeling
- **Tableau**: Interactive dashboard visualization
- **Kaggle**: Data source and community resources
- **Libraries**: 
  - scikit-learn (clustering, modeling)
  - pandas (data manipulation)
  - numpy (numerical computing)
  - matplotlib/seaborn (visualization)
  - scipy (statistical analysis)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Mental-health
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data directories:
```bash
mkdir -p data/raw data/processed data/outputs
```

## Usage

### Quick Start with Kaggle Dataset

If you have the Kaggle Student Mental Health dataset:

```bash
# Install dependencies
pip install -r requirements.txt

# Adapt and analyze Kaggle dataset
python src/run_kaggle_analysis.py
```

See [KAGGLE_DATASET_USAGE.md](KAGGLE_DATASET_USAGE.md) for detailed instructions.

### Using Your Own Data

### Data Preprocessing
```bash
python src/data_preprocessing.py
```

### Clustering Analysis
```bash
python src/clustering.py
```

### Correlation Analysis
```bash
python src/correlation_analysis.py
```

### Statistical Modeling
```bash
python src/statistical_modeling.py
```

### Generate Visualizations
```bash
python src/visualization.py
```

## Key Findings

1. **Peak Stress Periods**: Identified 5 high-risk academic weeks during midterms and finals
2. **At-Risk Groups**: Classified 3 distinct student groups requiring different intervention strategies
3. **Predictive Patterns**: Established correlations between academic workload and mental health indicators

## Methodology

See `docs/methodology.md` for detailed information about:
- Data collection and preprocessing
- Clustering algorithms (K-means, DBSCAN)
- Correlation analysis techniques
- Statistical modeling approaches
- Validation methods

## Dashboard

The interactive Tableau dashboard provides:
- Real-time stress level monitoring
- Weekly and monthly trend analysis
- Student group classification visualization
- Risk assessment metrics

## Future Enhancements

- Real-time data integration
- Machine learning model deployment
- Automated alert system for high-risk students
- Integration with campus wellness programs

## License

[Specify your license]

## Contact

[Your contact information]
