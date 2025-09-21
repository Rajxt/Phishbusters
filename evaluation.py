#evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class ModelEvaluator:
    def __init__(self):
        self.models = None
    
    def load_models(self):
        """Load trained models"""
        with open('trained_models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        return self.models
    
    def evaluate_comprehensive(self, X_test, y_test):
        """Comprehensive evaluation of the system"""
        print("="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        
        # Individual model predictions
        nb_pred = self.models['nb_model'].predict(X_test)
        lr_pred = self.models['lr_model'].predict(X_test)
        
        # Ensemble predictions
        ensemble_pred, ensemble_conf = self.smart_ensemble_predict(X_test)
        
        # Calculate metrics for all models
        models_results = {
            'Naive Bayes': {
                'predictions': nb_pred,
                'f1': f1_score(y_test, nb_pred),
                'precision': precision_score(y_test, nb_pred),
                'recall': recall_score(y_test, nb_pred),
                'fpr': self.calculate_false_positive_rate(y_test, nb_pred)
            },
            'Logistic Regression': {
                'predictions': lr_pred,
                'f1': f1_score(y_test, lr_pred),
                'precision': precision_score(y_test, lr_pred),
                'recall': recall_score(y_test, lr_pred),
                'fpr': self.calculate_false_positive_rate(y_test, lr_pred)
            },
            'Smart Ensemble': {
                'predictions': ensemble_pred,
                'f1': f1_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred),
                'recall': recall_score(y_test, ensemble_pred),
                'fpr': self.calculate_false_positive_rate(y_test, ensemble_pred),
                'avg_confidence': ensemble_conf.mean()
            }
        }
        
        # Print results
        print(f"{'Model':<20} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'FPR':<10}")
        print("-" * 60)
        for model_name, results in models_results.items():
            print(f"{model_name:<20} {results['f1']:<10.3f} {results['precision']:<10.3f} {results['recall']:<10.3f} {results['fpr']:<10.3f}")
        
        print(f"\nSmart Ensemble Average Confidence: {models_results['Smart Ensemble']['avg_confidence']:.3f}")
        
        # Show improvement
        ensemble_f1 = models_results['Smart Ensemble']['f1']
        best_individual_f1 = max(models_results['Naive Bayes']['f1'], models_results['Logistic Regression']['f1'])
        improvement = ((ensemble_f1 - best_individual_f1) / best_individual_f1) * 100
        
        print(f"Ensemble improvement over best individual model: +{improvement:.1f}%")
        
        return models_results
    
    def smart_ensemble_predict(self, X):
        """Smart ensemble prediction"""
        nb_proba = self.models['nb_model'].predict_proba(X)[:, 1]
        lr_proba = self.models['lr_model'].predict_proba(X)[:, 1]
        threshold = self.models['best_threshold']
        
        predictions = []
        confidences = []
        
        for nb_prob, lr_prob in zip(nb_proba, lr_proba):
            nb_confidence = abs(nb_prob - 0.5) * 2
            lr_confidence = abs(lr_prob - 0.5) * 2
            
            if (nb_confidence > threshold and lr_confidence > threshold and
                (nb_prob > 0.5) == (lr_prob > 0.5)):
                final_pred = 1 if nb_prob > 0.5 else 0
                final_conf = max(nb_confidence, lr_confidence)
            elif nb_confidence > lr_confidence:
                final_pred = 1 if nb_prob > 0.5 else 0
                final_conf = nb_confidence
            else:
                final_pred = 1 if lr_prob > 0.5 else 0
                final_conf = lr_confidence
            
            predictions.append(final_pred)
            confidences.append(final_conf)
        
        return np.array(predictions), np.array(confidences)
    
    def calculate_false_positive_rate(self, y_true, y_pred):
        """Calculate false positive rate"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Individual predictions
        nb_pred = self.models['nb_model'].predict(X_test)
        lr_pred = self.models['lr_model'].predict(X_test)
        ensemble_pred, _ = self.smart_ensemble_predict(X_test)
        
        predictions = [nb_pred, lr_pred, ensemble_pred]
        titles = ['Naive Bayes', 'Logistic Regression', 'Smart Ensemble']
        
        for i, (pred, title) in enumerate(zip(predictions, titles)):
            cm = confusion_matrix(y_test, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.show()

if __name__ == "__main__":
    # Load test data and models
    with open('data_with_trust_scores.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Prepare test data (you'd normally load this separately)
    from model_training import PhishingModelTrainer
    trainer = PhishingModelTrainer()
    X, _, _, _ = trainer.prepare_features(df)
    
    # Create test labels (adjust as needed)
    if 'label' in df.columns:
        y = df['label'].apply(lambda x: 1 if str(x).lower() in ['phishing', '1', 'spam'] else 0)
    else:
        y = np.random.choice([0, 1], size=len(df))
    
    # Use last 20% as test set
    test_size = int(0.2 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Evaluate
    evaluator = ModelEvaluator()
    evaluator.load_models()
    results = evaluator.evaluate_comprehensive(X_test, y_test)
    evaluator.plot_confusion_matrices(X_test, y_test)