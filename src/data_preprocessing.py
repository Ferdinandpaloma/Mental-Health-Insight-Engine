"""
Data Preprocessing Module
Handles data cleaning, transformation, and preparation for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses student mental health data for analysis"""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to raw data file
        """
        self.data_path = data_path
        self.processed_data = None
        
    def load_data(self, file_path):
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        print(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        print(f"Removed duplicates: {len(df) - len(df_clean)} records")
        
        # Handle missing values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Handle missing values in categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isna().sum() > 0:
                df_clean[col].fillna('Unknown', inplace=True)
        
        # Remove outliers using IQR method for stress-related columns
        stress_cols = [col for col in df_clean.columns if 'stress' in col.lower() or 'score' in col.lower()]
        for col in stress_cols:
            if col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Final cleaned dataset: {len(df_clean)} records")
        return df_clean
    
    def engineer_features(self, df):
        """
        Create new features for analysis
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_feat = df.copy()
        
        # Create academic week feature if date column exists
        date_cols = [col for col in df_feat.columns if 'date' in col.lower() or 'week' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df_feat[date_col] = pd.to_datetime(df_feat[date_col], errors='coerce')
            df_feat['academic_week'] = df_feat[date_col].dt.isocalendar().week
        
        # Create composite stress score if multiple stress indicators exist
        stress_cols = [col for col in df_feat.columns if 'stress' in col.lower()]
        if len(stress_cols) > 1:
            df_feat['composite_stress_score'] = df_feat[stress_cols].mean(axis=1)
        
        # Create risk level categories
        if 'composite_stress_score' in df_feat.columns:
            df_feat['risk_level'] = pd.cut(
                df_feat['composite_stress_score'],
                bins=[0, 3, 6, 10],
                labels=['Low', 'Medium', 'High']
            )
        
        return df_feat
    
    def normalize_features(self, df, columns=None):
        """
        Normalize features for clustering and modeling
        
        Args:
            df: DataFrame
            columns: List of columns to normalize (default: all numeric)
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns
        
        # Standard normalization (z-score)
        for col in columns:
            if col in df_norm.columns:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                if std > 0:
                    df_norm[f'{col}_normalized'] = (df_norm[col] - mean) / std
        
        return df_norm
    
    def preprocess_pipeline(self, file_path, save_path=None):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to raw data file
            save_path: Path to save processed data (optional)
            
        Returns:
            Processed DataFrame
        """
        # Load data
        df = self.load_data(file_path)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Normalize features
        df_processed = self.normalize_features(df_features)
        
        self.processed_data = df_processed
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}")
        
        return df_processed


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Update with your actual data path
    # processed_df = preprocessor.preprocess_pipeline(
    #     file_path='data/raw/student_data.csv',
    #     save_path='data/processed/processed_data.csv'
    # )
    
    print("Data preprocessing module ready!")
