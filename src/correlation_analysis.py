"""
Correlation Analysis Module
Identifies relationships between academic factors and mental health indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Analyzes correlations between variables to detect stress patterns"""
    
    def __init__(self):
        """Initialize correlation analyzer"""
        self.correlation_matrix = None
        self.significant_correlations = None
        
    def calculate_correlations(self, df, method='pearson', threshold=0.3):
        """
        Calculate correlation matrix
        
        Args:
            df: DataFrame with numeric columns
            method: Correlation method ('pearson' or 'spearman')
            threshold: Minimum absolute correlation to consider significant
            
        Returns:
            Correlation matrix
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if method == 'pearson':
            self.correlation_matrix = numeric_df.corr(method='pearson')
        elif method == 'spearman':
            self.correlation_matrix = numeric_df.corr(method='spearman')
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        # Find significant correlations
        self._find_significant_correlations(threshold)
        
        return self.correlation_matrix
    
    def _find_significant_correlations(self, threshold=0.3):
        """
        Identify significant correlations above threshold
        
        Args:
            threshold: Minimum absolute correlation value
        """
        corr_pairs = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    corr_pairs.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        self.significant_correlations = pd.DataFrame(corr_pairs)
        if len(self.significant_correlations) > 0:
            self.significant_correlations = self.significant_correlations.sort_values(
                'abs_correlation', ascending=False
            )
    
    def test_correlation_significance(self, df, var1, var2, method='pearson'):
        """
        Test statistical significance of correlation
        
        Args:
            df: DataFrame
            var1: First variable name
            var2: Second variable name
            method: Correlation method
            
        Returns:
            Correlation coefficient and p-value
        """
        if var1 not in df.columns or var2 not in df.columns:
            raise ValueError(f"Variables {var1} or {var2} not found in dataframe")
        
        data1 = df[var1].dropna()
        data2 = df[var2].dropna()
        
        # Align data
        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx]
        data2 = data2.loc[common_idx]
        
        if method == 'pearson':
            corr, p_value = pearsonr(data1, data2)
        elif method == 'spearman':
            corr, p_value = spearmanr(data1, data2)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        return corr, p_value
    
    def identify_stress_patterns(self, df, stress_columns=None, academic_columns=None):
        """
        Identify correlations between academic factors and stress indicators
        
        Args:
            df: DataFrame with student data
            stress_columns: List of stress-related column names
            academic_columns: List of academic-related column names
            
        Returns:
            DataFrame with stress-academic correlations
        """
        if stress_columns is None:
            stress_columns = [col for col in df.columns 
                            if any(keyword in col.lower() for keyword in 
                                  ['stress', 'anxiety', 'depression', 'mental'])]
        
        if academic_columns is None:
            academic_columns = [col for col in df.columns 
                             if any(keyword in col.lower() for keyword in 
                                   ['grade', 'gpa', 'course', 'assignment', 'exam', 'workload'])]
        
        stress_patterns = []
        
        for stress_col in stress_columns:
            for academic_col in academic_columns:
                if stress_col in df.columns and academic_col in df.columns:
                    try:
                        corr, p_value = self.test_correlation_significance(
                            df, stress_col, academic_col
                        )
                        
                        stress_patterns.append({
                            'stress_indicator': stress_col,
                            'academic_factor': academic_col,
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'strength': 'strong' if abs(corr) > 0.7 else 
                                       'moderate' if abs(corr) > 0.4 else 'weak'
                        })
                    except Exception as e:
                        print(f"Error calculating correlation between {stress_col} and {academic_col}: {e}")
        
        patterns_df = pd.DataFrame(stress_patterns)
        if len(patterns_df) > 0:
            patterns_df = patterns_df.sort_values('correlation', key=abs, ascending=False)
        
        return patterns_df
    
    def detect_peak_stress_periods(self, df, date_column=None, stress_column=None):
        """
        Detect peak stress periods across academic calendar
        
        Args:
            df: DataFrame with temporal data
            date_column: Name of date/week column
            stress_column: Name of stress indicator column
            
        Returns:
            DataFrame with weekly stress averages and peak periods
        """
        if date_column is None:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
            date_column = date_cols[0] if date_cols else None
        
        if stress_column is None:
            stress_cols = [col for col in df.columns if 'stress' in col.lower()]
            stress_column = stress_cols[0] if stress_cols else None
        
        if date_column is None or stress_column is None:
            raise ValueError("Date column and stress column must be specified")
        
        # Group by week/period
        if 'week' in date_column.lower():
            period_col = date_column
        else:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df['academic_week'] = df[date_column].dt.isocalendar().week
            period_col = 'academic_week'
        
        weekly_stress = df.groupby(period_col)[stress_column].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        weekly_stress.columns = [period_col, 'avg_stress', 'median_stress', 'std_stress', 'student_count']
        
        # Identify top 5 high-risk weeks
        weekly_stress = weekly_stress.sort_values('avg_stress', ascending=False)
        weekly_stress['risk_rank'] = range(1, len(weekly_stress) + 1)
        weekly_stress['high_risk'] = weekly_stress['risk_rank'] <= 5
        
        return weekly_stress
    
    def visualize_correlation_matrix(self, save_path=None, figsize=(12, 10)):
        """
        Visualize correlation matrix as heatmap
        
        Args:
            save_path: Path to save visualization
            figsize: Figure size
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Run calculate_correlations() first.")
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation Matrix: Academic Factors vs Mental Health Indicators', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def visualize_stress_patterns(self, weekly_stress_df, save_path=None):
        """
        Visualize stress patterns over time
        
        Args:
            weekly_stress_df: DataFrame from detect_peak_stress_periods()
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        period_col = weekly_stress_df.columns[0]
        
        # Line plot of average stress over time
        axes[0].plot(weekly_stress_df[period_col], weekly_stress_df['avg_stress'], 
                    marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0].fill_between(weekly_stress_df[period_col], 
                           weekly_stress_df['avg_stress'] - weekly_stress_df['std_stress'],
                           weekly_stress_df['avg_stress'] + weekly_stress_df['std_stress'],
                           alpha=0.3, color='steelblue')
        axes[0].set_xlabel('Academic Week', fontsize=12)
        axes[0].set_ylabel('Average Stress Level', fontsize=12)
        axes[0].set_title('Stress Levels Across Academic Calendar', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight high-risk weeks
        high_risk_weeks = weekly_stress_df[weekly_stress_df['high_risk']]
        axes[0].scatter(high_risk_weeks[period_col], high_risk_weeks['avg_stress'],
                       color='red', s=200, marker='*', zorder=5, label='High-Risk Weeks')
        axes[0].legend()
        
        # Bar chart of top 5 high-risk weeks
        top_5 = weekly_stress_df.head(5)
        axes[1].barh(range(len(top_5)), top_5['avg_stress'], color='crimson', alpha=0.7)
        axes[1].set_yticks(range(len(top_5)))
        axes[1].set_yticklabels([f"Week {int(w)}" for w in top_5[period_col]])
        axes[1].set_xlabel('Average Stress Level', fontsize=12)
        axes[1].set_title('Top 5 High-Risk Academic Weeks', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stress pattern visualization saved to {save_path}")
        
        plt.show()
    
    def generate_correlation_report(self, df, save_path=None):
        """
        Generate comprehensive correlation analysis report
        
        Args:
            df: DataFrame with student data
            save_path: Path to save report
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate correlations
        corr_matrix = self.calculate_correlations(df)
        
        # Identify stress patterns
        stress_patterns = self.identify_stress_patterns(df)
        
        # Detect peak periods
        weekly_stress = self.detect_peak_stress_periods(df)
        
        report = {
            'correlation_matrix': corr_matrix,
            'significant_correlations': self.significant_correlations,
            'stress_patterns': stress_patterns,
            'peak_stress_periods': weekly_stress,
            'top_5_high_risk_weeks': weekly_stress.head(5)
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')
                if self.significant_correlations is not None:
                    self.significant_correlations.to_excel(writer, sheet_name='Significant Correlations', index=False)
                if len(stress_patterns) > 0:
                    stress_patterns.to_excel(writer, sheet_name='Stress Patterns', index=False)
                weekly_stress.to_excel(writer, sheet_name='Weekly Stress Analysis', index=False)
            
            print(f"Correlation report saved to {save_path}")
        
        return report


if __name__ == "__main__":
    # Example usage
    analyzer = CorrelationAnalyzer()
    
    # Example with sample data
    # df = pd.read_csv('data/processed/processed_data.csv')
    # report = analyzer.generate_correlation_report(df, save_path='data/outputs/correlation_report.xlsx')
    
    print("Correlation analysis module ready!")
