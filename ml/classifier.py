"""Machine Learning Classification for ATS Prediction

Demonstrates:
- Deep learning architecture
- Neural networks
- Supervised learning
- Predictive analytics
- Hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict


class ATSPredictor:
    """ML classifier for ATS success prediction
    
    Implements:
    - Machine learning classification
    - Feature engineering
    - Model selection
    - Cross-validation
    """
    
    def __init__(self, model_type: str = "neural_network"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model with hyperparameter tuning"""
        if self.model_type == "neural_network":
            # Neural network with backpropagation
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',  # Adam optimizer
                learning_rate='adaptive',
                max_iter=1000
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
    
    def feature_engineering(self, resume: Dict, job: Dict) -> np.ndarray:
        """Automated feature engineering
        
        Creates features:
        - Keyword match ratio
        - Skill overlap
        - Experience years
        - Education match
        - Salary alignment
        """
        features = []
        
        # Keyword similarity (cosine)
        features.append(self._keyword_match(resume, job))
        
        # Skill overlap ratio
        features.append(self._skill_overlap(resume, job))
        
        # Experience match
        features.append(self._experience_match(resume, job))
        
        # Education level
        features.append(self._education_score(resume, job))
        
        # Salary fit
        features.append(self._salary_alignment(resume, job))
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train classifier with cross-validation"""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='roc_auc'
        )
        
        # Train model
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))
        
        return grid_search.best_params_
    
    def predict_ats_success(self, resume: Dict, job: Dict) -> Tuple[float, Dict]:
        """Predict ATS pass probability
        
        Returns:
        - Pass rate (0-1)
        - Feature importance
        """
        features = self.feature_engineering(resume, job)
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        proba = self.model.predict_proba(features_scaled)[0][1]
        
        # Feature importance (for explainable AI)
        importance = self._get_feature_importance()
        
        return proba, importance
    
    def _keyword_match(self, resume: Dict, job: Dict) -> float:
        """Calculate keyword overlap"""
        resume_keywords = set(resume.get("keywords", []))
        job_keywords = set(job.get("keywords", []))
        
        if not job_keywords:
            return 0.0
        
        return len(resume_keywords & job_keywords) / len(job_keywords)
    
    def _skill_overlap(self, resume: Dict, job: Dict) -> float:
        """Calculate skill match ratio"""
        resume_skills = set(resume.get("skills", {}).keys())
        job_skills = set(job.get("required_skills", []))
        
        if not job_skills:
            return 0.0
        
        return len(resume_skills & job_skills) / len(job_skills)
    
    def _experience_match(self, resume: Dict, job: Dict) -> float:
        """Match experience years"""
        resume_exp = resume.get("years_experience", 0)
        required_exp = job.get("min_experience", 0)
        
        return min(resume_exp / max(required_exp, 1), 1.0)
    
    def _education_score(self, resume: Dict, job: Dict) -> float:
        """Education level match"""
        education_levels = {"HS": 1, "BS": 2, "MS": 3, "PhD": 4}
        resume_edu = education_levels.get(resume.get("education", "BS"), 2)
        required_edu = education_levels.get(job.get("required_education", "BS"), 2)
        
        return min(resume_edu / required_edu, 1.0)
    
    def _salary_alignment(self, resume: Dict, job: Dict) -> float:
        """Salary expectations alignment"""
        expected = resume.get("expected_salary", 0)
        offered = job.get("salary_max", 0)
        
        if offered == 0:
            return 0.5
        
        return min(expected / offered, 1.0)
    
    def _get_feature_importance(self) -> Dict:
        """Extract feature importance for explainable AI"""
        if hasattr(self.model, 'feature_importances_'):
            return {
                "keyword_match": self.model.feature_importances_[0],
                "skill_overlap": self.model.feature_importances_[1],
                "experience": self.model.feature_importances_[2],
                "education": self.model.feature_importances_[3],
                "salary": self.model.feature_importances_[4]
            }
        return {}


class DeepNeuralNetwork(nn.Module):
    """Deep learning architecture for job matching
    
    Implements:
    - Deep neural networks
    - Backpropagation
    - Gradient descent
    - Dropout regularization
    - Batch normalization
    """
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build deep architecture
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Batch normalization
                nn.ReLU(),  # Activation function
                nn.Dropout(0.3)  # Dropout regularization
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)
    
    def train_model(self, X_train, y_train, epochs: int = 100):
        """Train with backpropagation and gradient descent"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
