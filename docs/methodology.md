# Methodology: Campus Mental Health Insight Engine

## Overview

This document outlines the methodology used in the Campus Mental Health Insight Engine project, detailing the data analysis approaches, algorithms, and validation methods employed to identify at-risk student groups and peak stress periods.

## 1. Data Collection and Preprocessing

### 1.1 Data Source
- **Dataset**: 10,000+ simulated student data points
- **Source**: Kaggle database
- **Time Period**: Academic year data covering multiple semesters

### 1.2 Data Preprocessing Steps

#### Data Cleaning
1. **Duplicate Removal**: Identified and removed duplicate records
2. **Missing Value Handling**:
   - Numeric columns: Median imputation
   - Categorical columns: Mode imputation or "Unknown" category
3. **Outlier Detection**: Interquartile Range (IQR) method for stress-related variables
   - Removed values beyond Q1 - 1.5×IQR and Q3 + 1.5×IQR

#### Feature Engineering
1. **Temporal Features**:
   - Academic week calculation from date columns
   - Semester period identification
   - Peak period flags (midterms, finals)

2. **Composite Scores**:
   - Composite stress score from multiple stress indicators
   - Risk level categorization (Low, Medium, High)

3. **Normalization**:
   - Z-score normalization for clustering algorithms
   - Min-max scaling for specific models

## 2. Clustering Analysis

### 2.1 Objective
Identify distinct student groups based on mental health and academic patterns.

### 2.2 Algorithms Used

#### K-Means Clustering
- **Purpose**: Partition students into k distinct groups
- **Method**: 
  - Optimal k determined using elbow method and silhouette score
  - Standardized features before clustering
  - Multiple random initializations (n_init=10)

#### DBSCAN (Density-Based Clustering)
- **Purpose**: Identify density-based clusters and outliers
- **Parameters**:
  - eps: Maximum distance between samples
  - min_samples: Minimum samples in neighborhood

#### Hierarchical Clustering
- **Purpose**: Create hierarchical cluster structure
- **Linkage**: Ward's method for minimizing variance

### 2.3 Cluster Validation
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Lower values indicate better clustering
- **Visual Inspection**: 2D/3D scatter plots of clusters

### 2.4 Results
- Identified **3 distinct at-risk student groups**:
  1. **High Academic Stress Group**: High workload, moderate stress
  2. **Chronic Stress Group**: Consistently high stress across all periods
  3. **At-Risk Transition Group**: Escalating stress patterns

## 3. Correlation Analysis

### 3.1 Objective
Detect relationships between academic factors and mental health indicators.

### 3.2 Methods

#### Pearson Correlation
- Measures linear relationships between continuous variables
- Range: -1 to +1
- Significance testing with p-values

#### Spearman Correlation
- Measures monotonic relationships
- Used for ordinal or non-normal data

### 3.3 Key Findings
- Strong positive correlation between academic workload and stress levels
- Negative correlation between sleep hours and stress indicators
- Moderate correlation between GPA and mental health scores

### 3.4 Peak Stress Period Detection
- **Method**: Weekly aggregation of stress indicators
- **Identification**: Top 5 weeks with highest average stress
- **Validation**: Statistical significance testing

## 4. Statistical Modeling

### 4.1 Objective
Build predictive models to classify students into risk categories.

### 4.2 Target Variable Creation
- **Method 1 - Threshold**: Top 25% stress scores classified as at-risk
- **Method 2 - Percentile**: Three-tier classification (Low/Medium/High)
- **Method 3 - Cluster-based**: Using cluster labels as risk indicators

### 4.3 Models Implemented

#### Logistic Regression
- **Purpose**: Baseline classification model
- **Features**: Standardized numeric features
- **Regularization**: L2 regularization

#### Random Forest Classifier
- **Purpose**: Non-linear pattern detection
- **Parameters**:
  - n_estimators: 100 trees
  - Feature importance extraction
- **Advantages**: Handles non-linear relationships, feature importance

#### Gradient Boosting Classifier
- **Purpose**: Ensemble method for improved accuracy
- **Parameters**: n_estimators: 100
- **Advantages**: Sequential learning, high predictive power

### 4.4 Model Evaluation

#### Metrics Used
1. **Accuracy**: Overall classification correctness
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under receiver operating characteristic curve

#### Validation Strategy
- **Train-Test Split**: 80-20 split
- **Cross-Validation**: 5-fold cross-validation
- **Stratified Sampling**: Maintains class distribution

### 4.5 Feature Importance
- Extracted from Random Forest model
- Identified top predictors of at-risk status
- Used for model interpretation and intervention planning

## 5. Temporal Analysis

### 5.1 Peak Stress Period Modeling
- **Method**: Time series analysis of weekly stress averages
- **Identification**: Statistical ranking of academic weeks
- **Result**: Modeled 5 high-risk academic weeks

### 5.2 Pattern Recognition
- **Trend Analysis**: Identifying increasing/decreasing stress trends
- **Seasonal Patterns**: Detecting semester-specific patterns
- **Event Correlation**: Linking stress spikes to academic events

## 6. Visualization and Dashboard

### 6.1 Interactive Dashboard (Tableau)
- **Components**:
  - Stress timeline visualization
  - Cluster distribution charts
  - Risk group breakdowns
  - Correlation heatmaps
  - Weekly stress patterns

### 6.2 Key Visualizations
1. **Stress Timeline**: Line chart showing stress over academic calendar
2. **Cluster Scatter Plots**: 2D visualization of student groups
3. **Risk Distribution**: Pie/bar charts of risk categories
4. **Correlation Heatmap**: Matrix visualization of relationships
5. **Top Risk Weeks**: Bar chart of high-risk periods

## 7. Validation and Quality Assurance

### 7.1 Data Quality
- Data completeness checks
- Consistency validation
- Range validation for numeric variables

### 7.2 Model Validation
- Cross-validation scores
- Out-of-sample testing
- Performance metrics comparison across models

### 7.3 Result Validation
- Cluster interpretability review
- Correlation significance testing
- Temporal pattern validation against known academic calendar

## 8. Limitations and Assumptions

### 8.1 Data Limitations
- Simulated data may not capture all real-world complexities
- Missing contextual factors (social support, personal circumstances)
- Self-reported data potential biases

### 8.2 Methodological Assumptions
- Linear relationships in correlation analysis
- Stationarity in temporal patterns
- Independence of observations

### 8.3 Generalizability
- Results specific to dataset characteristics
- May require recalibration for different institutions
- Temporal patterns may vary by academic calendar

## 9. Ethical Considerations

### 9.1 Privacy
- Anonymized student data
- Aggregate reporting to protect individual privacy

### 9.2 Bias Mitigation
- Regular model auditing
- Diverse feature representation
- Fairness checks across student demographics

### 9.3 Intervention Ethics
- Proactive support, not punitive measures
- Student consent for interventions
- Transparent use of predictive models

## 10. Future Enhancements

### 10.1 Model Improvements
- Deep learning approaches for complex patterns
- Real-time data integration
- Multi-institutional validation

### 10.2 Feature Expansion
- Social media sentiment analysis
- Academic performance trends
- Engagement metrics

### 10.3 Deployment
- Automated alert systems
- Integration with student information systems
- Mobile application for students

## References

- Scikit-learn documentation for clustering and classification algorithms
- Statistical methods from scipy and statsmodels
- Best practices from academic mental health research
- Tableau visualization guidelines
