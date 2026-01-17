"""
Statistical Modeling Module
Builds models to identify and classify at-risk student groups
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AtRiskModeling:
    """Statistical modeling for at-risk student identification"""
    
    def __init__(self, random_state=42):
        """
        Initialize modeling class
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        
    def prepare_target_variable(self, df, target_column=None, method='threshold'):
        """
        Create target variable for at-risk classification
        
        Args:
            df: DataFrame with student data
            target_column: Column to use for target (stress score, etc.)
            method: Method to create binary target ('threshold', 'percentile', 'cluster')
            
        Returns:
            DataFrame with target variable
        """
        df_model = df.copy()
        
        if target_column is None:
            # Auto-detect stress-related column
            stress_cols = [col for col in df.columns if 'stress' in col.lower()]
            target_column = stress_cols[0] if stress_cols else None
        
        if target_column is None:
            raise ValueError("Target column must be specified or detectable")
        
        if method == 'threshold':
            # Use fixed threshold (e.g., stress score > 7 = at-risk)
            threshold = df[target_column].quantile(0.75)  # Top 25% as at-risk
            df_model['at_risk'] = (df[target_column] >= threshold).astype(int)
        
        elif method == 'percentile':
            # Use percentile-based classification
            p75 = df[target_column].quantile(0.75)
            p50 = df[target_column].quantile(0.50)
            df_model['at_risk'] = pd.cut(
                df[target_column],
                bins=[-np.inf, p50, p75, np.inf],
                labels=[0, 1, 2]  # Low, Medium, High risk
            ).astype(int)
        
        elif method == 'cluster':
            # Use existing cluster labels if available
            if 'cluster' in df.columns:
                # Assume higher cluster IDs or specific clusters are at-risk
                df_model['at_risk'] = (df['cluster'] >= df['cluster'].median()).astype(int)
            else:
                raise ValueError("Cluster column not found. Run clustering first or use different method.")
        
        return df_model
    
    def select_features(self, df, exclude_columns=None):
        """
        Select features for modeling
        
        Args:
            df: DataFrame with features
            exclude_columns: Columns to exclude from features
            
        Returns:
            List of feature column names
        """
        if exclude_columns is None:
            exclude_columns = ['at_risk', 'cluster', 'student_id', 'id']
        
        # Select numeric features
        feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in exclude_columns]
        
        # Filter out columns with too many missing values (>50%)
        feature_columns = [col for col in feature_columns 
                          if df[col].notna().sum() / len(df) > 0.5]
        
        return feature_columns
    
    def train_models(self, X, y, models_to_train=None):
        """
        Train multiple classification models
        
        Args:
            X: Feature matrix
            y: Target variable
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['logistic', 'random_forest', 'gradient_boosting']
        
        # Handle missing values before train/test split
        X = X.fillna(X.median())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        trained_models = {}
        
        # Logistic Regression
        if 'logistic' in models_to_train:
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            trained_models['logistic'] = {
                'model': lr,
                'X_test': X_test_scaled,
                'y_test': y_test,
                'X_train': X_train_scaled,
                'y_train': y_train
            }
        
        # Random Forest
        if 'random_forest' in models_to_train:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            trained_models['random_forest'] = {
                'model': rf,
                'X_test': X_test,
                'y_test': y_test,
                'X_train': X_train,
                'y_train': y_train
            }
        
        # Gradient Boosting
        if 'gradient_boosting' in models_to_train:
            gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
            gb.fit(X_train, y_train)
            trained_models['gradient_boosting'] = {
                'model': gb,
                'X_test': X_test,
                'y_test': y_test,
                'X_train': X_train,
                'y_train': y_train
            }
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.models:
            raise ValueError("No models trained. Run train_models() first.")
        
        results = []
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # ROC AUC
            roc_auc = None
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
            # Cross-validation score
            cv_scores = cross_val_score(model, model_data['X_train'], model_data['y_train'], 
                                       cv=5, scoring='roc_auc' if roc_auc else 'accuracy')
            
            results.append({
                'model': model_name,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            })
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def get_feature_importance(self, model_name='random_forest', top_n=10):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name: Name of model to extract importance from
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]['model']
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not support feature importance")
        
        # Get feature names (assuming they were passed as DataFrame columns)
        # This would need to be adjusted based on actual implementation
        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        self.feature_importance[model_name] = importance_df
        return importance_df
    
    def classify_student_groups(self, df, model_name='random_forest', feature_columns=None):
        """
        Classify students into risk groups using trained model
        
        Args:
            df: DataFrame with student data
            model_name: Model to use for classification
            feature_columns: Features to use for prediction
            
        Returns:
            DataFrame with risk classifications
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train models first.")
        
        model = self.models[model_name]['model']
        
        if feature_columns is None:
            feature_columns = self.select_features(df)
        
        X = df[feature_columns]
        
        # Scale if needed (for logistic regression)
        if model_name == 'logistic':
            X = self.scaler.transform(X)
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        df_classified = df.copy()
        df_classified['predicted_risk'] = predictions
        df_classified['risk_probability'] = probabilities[:, 1] if probabilities is not None else None
        
        # Create risk groups
        if probabilities is not None:
            df_classified['risk_group'] = pd.cut(
                probabilities[:, 1],
                bins=[0, 0.33, 0.67, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
        
        return df_classified
    
    def visualize_model_performance(self, save_path=None):
        """
        Visualize model performance metrics
        
        Args:
            save_path: Path to save visualization
        """
        if not self.models:
            raise ValueError("No models to visualize. Train models first.")
        
        # Get evaluation results
        results = self.evaluate_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(results['model'], results['accuracy'], color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # ROC AUC comparison
        if results['roc_auc'].notna().any():
            axes[0, 1].bar(results['model'], results['roc_auc'], color='crimson', alpha=0.7)
            axes[0, 1].set_ylabel('ROC AUC Score', fontsize=12)
            axes[0, 1].set_title('ROC AUC Comparison', fontsize=14, fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # F1 Score comparison
        axes[1, 0].bar(results['model'], results['f1_score'], color='green', alpha=0.7)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Cross-validation scores
        x_pos = np.arange(len(results))
        axes[1, 1].bar(x_pos, results['cv_mean'], yerr=results['cv_std'], 
                      color='orange', alpha=0.7, capsize=5)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(results['model'])
        axes[1, 1].set_ylabel('CV Score (Mean ± Std)', fontsize=12)
        axes[1, 1].set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model performance visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_confusion_matrix(self, model_name='random_forest', save_path=None):
        """
        Visualize confusion matrix for a specific model
        
        Args:
            model_name: Name of model to visualize
            save_path: Path to save visualization
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        model = model_data['model']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not At-Risk', 'At-Risk'],
                   yticklabels=['Not At-Risk', 'At-Risk'])
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title(f'Confusion Matrix: {model_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def modeling_pipeline(self, df, target_column=None, feature_columns=None, 
                         save_results=True):
        """
        Complete modeling pipeline
        
        Args:
            df: DataFrame with student data
            target_column: Target variable column
            feature_columns: Feature columns to use
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with models, predictions, and metrics
        """
        # Prepare target variable
        df_model = self.prepare_target_variable(df, target_column)
        
        # Select features
        if feature_columns is None:
            feature_columns = self.select_features(df_model)
        
        X = df_model[feature_columns]
        y = df_model['at_risk']
        
        # Train models
        print("Training models...")
        self.train_models(X, y)
        
        # Evaluate models
        print("Evaluating models...")
        evaluation_results = self.evaluate_models()
        print("\nModel Performance:")
        print(evaluation_results.to_string())
        
        # Get feature importance
        if 'random_forest' in self.models:
            print("\nTop Features (Random Forest):")
            importance = self.get_feature_importance('random_forest')
            print(importance.to_string())
        
        # Classify students
        print("\nClassifying students...")
        df_classified = self.classify_student_groups(df_model, feature_columns=feature_columns)
        
        # Save results
        if save_results:
            output_dir = Path('data/outputs')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            evaluation_results.to_csv(output_dir / 'model_evaluation.csv', index=False)
            df_classified.to_csv(output_dir / 'student_classifications.csv', index=False)
            
            if 'random_forest' in self.models:
                importance.to_csv(output_dir / 'feature_importance.csv', index=False)
            
            print(f"\nResults saved to {output_dir}")
        
        return {
            'models': self.models,
            'evaluation': evaluation_results,
            'classifications': df_classified,
            'feature_importance': self.feature_importance
        }


if __name__ == "__main__":
    # Example usage
    modeler = AtRiskModeling()
    
    # Example with sample data
    # df = pd.read_csv('data/processed/processed_data.csv')
    # results = modeler.modeling_pipeline(df)
    
    print("Statistical modeling module ready!")
