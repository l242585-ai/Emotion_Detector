"""
Baseline Model: TF-IDF + Logistic Regression
Simple but effective baseline for emotion classification.
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


class BaselineModel:
    """
    Baseline emotion classifier using TF-IDF features and Logistic Regression.
    
    Features:
        - TF-IDF vectorization with bigrams
        - Logistic Regression with balanced class weights
        - L2 regularization (default)
    
    Args:
        max_features (int): Maximum number of TF-IDF features (default: 5000)
        ngram_range (tuple): N-gram range for TF-IDF (default: (1,2) for unigrams and bigrams)
        max_iter (int): Maximum iterations for LogReg (default: 1000)
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), max_iter=1000):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            strip_accents='unicode'
        )
        
        # Logistic Regression Classifier
        self.classifier = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=max_iter,
            multi_class='multinomial',  # For 6-class classification
            solver='lbfgs',
            random_state=42
        )
        
        self.is_fitted = False
        self.classes_ = None
    
    def fit(self, X_train, y_train):
        """
        Train the baseline model.
        
        Args:
            X_train (list or array): Training texts
            y_train (array): Training labels (0-5)
        
        Returns:
            self
        """
        print(f"Vectorizing training data with TF-IDF...")
        print(f"  - Max features: {self.max_features}")
        print(f"  - N-gram range: {self.ngram_range}")
        
        # Transform text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  - Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train classifier
        print(f"\nTraining Logistic Regression...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        
        print(f"✓ Model training complete!")
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test (list or array): Test texts
        
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Transform test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Predict
        predictions = self.classifier.predict(X_test_tfidf)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Args:
            X_test (list or array): Test texts
        
        Returns:
            np.ndarray: Probability for each class, shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        probabilities = self.classifier.predict_proba(X_test_tfidf)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test (list or array): Test texts
            y_test (array): True labels
        
        Returns:
            dict: Dictionary with accuracy and F1 scores
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        f1_weighted = f1_score(y_test, predictions, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': classification_report(y_test, predictions)
        }
        
        return results
    
    def save(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Train the model first.")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_iter': self.max_iter,
            'classes_': self.classes_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        
        Returns:
            BaselineModel: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        model = cls(
            max_features=model_data['max_features'],
            ngram_range=model_data['ngram_range'],
            max_iter=model_data['max_iter']
        )
        
        # Restore fitted components
        model.vectorizer = model_data['vectorizer']
        model.classifier = model_data['classifier']
        model.classes_ = model_data['classes_']
        model.is_fitted = True
        
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, top_n=10):
        """
        Get top features (words) for each emotion class.
        
        Args:
            top_n (int): Number of top features to return per class
        
        Returns:
            dict: Dictionary mapping class index to top features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Get coefficients for each class
        importance = {}
        for idx, class_label in enumerate(self.classes_):
            # Get coefficients for this class
            coef = self.classifier.coef_[idx]
            # Get indices of top N features
            top_indices = np.argsort(coef)[-top_n:][::-1]
            # Get feature names
            top_features = [(feature_names[i], coef[i]) for i in top_indices]
            importance[int(class_label)] = top_features
        
        return importance
