# Quick Start Guide

## Campus Mental Health Insight Engine

This guide will help you get started with the analysis pipeline quickly.

## Prerequisites

1. **Python 3.8+** installed
2. **pip** package manager

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd Mental-health
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start with Sample Data

1. **Generate sample data:**
```bash
python src/generate_sample_data.py
```

This will create a sample dataset with 10,000 student records in `data/raw/student_data.csv`.

2. **Run the complete analysis pipeline:**
```bash
python src/main.py data/raw/student_data.csv
```

This will:
- Preprocess the data
- Perform clustering analysis (identify 3 student groups)
- Analyze correlations
- Detect peak stress periods (top 5 high-risk weeks)
- Build statistical models
- Generate visualizations
- Export data for Tableau

## Using Your Own Data

1. **Prepare your data:**
   - Place your CSV or Excel file in `data/raw/`
   - Ensure your data includes columns related to:
     - Stress indicators (e.g., stress_score, anxiety_score)
     - Academic factors (e.g., gpa, course_load, exam_scores)
     - Temporal data (e.g., date, academic_week)

2. **Run the analysis:**
```bash
python src/main.py data/raw/your_data.csv
```

## Individual Module Usage

### Data Preprocessing
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(
    file_path='data/raw/student_data.csv',
    save_path='data/processed/processed_data.csv'
)
```

### Clustering Analysis
```python
from src.clustering import StudentClustering

cluster_analyzer = StudentClustering(n_clusters=3)
df_clustered, cluster_stats = cluster_analyzer.cluster_pipeline(
    df_processed,
    method='kmeans'
)
```

### Correlation Analysis
```python
from src.correlation_analysis import CorrelationAnalyzer

corr_analyzer = CorrelationAnalyzer()
report = corr_analyzer.generate_correlation_report(
    df_processed,
    save_path='data/outputs/correlation_report.xlsx'
)
```

### Statistical Modeling
```python
from src.statistical_modeling import AtRiskModeling

modeler = AtRiskModeling()
results = modeler.modeling_pipeline(df_clustered)
```

## Output Files

After running the analysis, you'll find:

- **Processed Data**: `data/processed/processed_data.csv`
- **Cluster Statistics**: `data/outputs/cluster_statistics.csv`
- **Correlation Report**: `data/outputs/correlation_report.xlsx`
- **Weekly Stress Analysis**: `data/outputs/weekly_stress_analysis.csv`
- **Model Evaluation**: `data/outputs/model_evaluation.csv`
- **Student Classifications**: `data/outputs/student_classifications.csv`
- **Visualizations**: `data/outputs/*.png` and `*.html`
- **Tableau Data**: `data/outputs/tableau/*.csv`

## Jupyter Notebooks

For interactive analysis, use the provided notebooks:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Tableau Integration

1. Export data using the visualization module:
```python
from src.visualization import DashboardVisualizer

visualizer = DashboardVisualizer()
visualizer.export_for_tableau(
    df_clustered,
    cluster_stats=cluster_stats,
    weekly_stress=weekly_stress,
    output_dir='data/outputs/tableau'
)
```

2. Open Tableau and connect to the CSV files in `data/outputs/tableau/`

## Troubleshooting

### Import Errors
If you encounter import errors, make sure you're running scripts from the project root directory:
```bash
cd /path/to/Mental-health
python src/main.py
```

### Missing Dependencies
If you get module not found errors:
```bash
pip install -r requirements.txt
```

### Data Format Issues
Ensure your data file:
- Is in CSV or Excel format
- Has proper column headers
- Contains numeric columns for analysis

## Next Steps

- Review the methodology in `docs/methodology.md`
- Customize the analysis parameters in each module
- Explore the Jupyter notebooks for interactive analysis
- Build your Tableau dashboard using the exported data

## Support

For questions or issues, refer to:
- README.md for project overview
- docs/methodology.md for detailed methodology
- Individual module docstrings for API documentation
