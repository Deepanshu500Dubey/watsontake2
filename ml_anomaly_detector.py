import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import warnings
from typing import Dict, List, Any, Optional
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, expenses_df: pd.DataFrame, current_expense: Dict) -> np.ndarray:
        """Prepare features for ML model without temporal features"""
        features = []
        feature_names = []
        
        # 1. Basic amount features
        features.extend([
            current_expense['amount'],
            np.log1p(current_expense['amount']),
            current_expense['amount'] ** 0.5  # Square root transform
        ])
        feature_names.extend(['amount', 'log_amount', 'sqrt_amount'])
        
        # 2. Employee behavior features
        employee_history = expenses_df[expenses_df['employee_id'] == current_expense['employee_id']]
        if len(employee_history) > 0:
            emp_amounts = employee_history['amount']
            features.extend([
                len(employee_history),
                emp_amounts.mean(),
                emp_amounts.std() if len(employee_history) > 1 else 0,
                emp_amounts.median(),
                emp_amounts.max(),
                (current_expense['amount'] - emp_amounts.mean()) / (emp_amounts.std() + 1e-6),
                (current_expense['amount'] - emp_amounts.median()) / (emp_amounts.std() + 1e-6),
            ])
            feature_names.extend([
                'emp_submission_count', 'emp_mean_amount', 'emp_std_amount',
                'emp_median_amount', 'emp_max_amount', 'emp_z_score_mean', 'emp_z_score_median'
            ])
        else:
            features.extend([1, current_expense['amount'], 0, current_expense['amount'], 
                           current_expense['amount'], 0, 0])
            feature_names.extend([
                'emp_submission_count', 'emp_mean_amount', 'emp_std_amount',
                'emp_median_amount', 'emp_max_amount', 'emp_z_score_mean', 'emp_z_score_median'
            ])
            
        # 3. Department behavior features
        dept_history = expenses_df[expenses_df['department'] == current_expense['department']]
        if len(dept_history) > 0:
            dept_amounts = dept_history['amount']
            features.extend([
                len(dept_history),
                dept_amounts.mean(),
                dept_amounts.std(),
                dept_amounts.median(),
                (current_expense['amount'] - dept_amounts.mean()) / (dept_amounts.std() + 1e-6),
                (current_expense['amount'] > dept_amounts.quantile(0.95)) * 1,
            ])
            feature_names.extend([
                'dept_submission_count', 'dept_mean_amount', 'dept_std_amount',
                'dept_median_amount', 'dept_z_score', 'dept_95th_percentile'
            ])
        else:
            features.extend([1, current_expense['amount'], 0, current_expense['amount'], 0, 0])
            feature_names.extend([
                'dept_submission_count', 'dept_mean_amount', 'dept_std_amount',
                'dept_median_amount', 'dept_z_score', 'dept_95th_percentile'
            ])
            
        # 4. Purpose-based features
        purpose_mapping = {
            'Travel': 1, 'Client Entertainment': 2, 'Meals': 3, 'Supplies': 4
        }
        purpose_value = purpose_mapping.get(current_expense['purpose'], 0)
        features.extend([purpose_value])
        feature_names.extend(['purpose_encoded'])
        
        # 5. Remove temporal features (day_of_month, day_of_week) since we don't have submission_date
        
        # 6. Interaction features
        features.extend([
            current_expense['amount'] * purpose_value,
            current_expense['amount'] / (dept_history['amount'].mean() + 1e-6) if len(dept_history) > 0 else 0
        ])
        feature_names.extend(['amount_purpose_interaction', 'amount_dept_ratio'])
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def train_model(self, expenses_df: pd.DataFrame) -> Dict[str, Any]:
        """Train the ML model on historical data"""
        if len(expenses_df) < 20:
            logger.warning("Insufficient data for training ML model")
            self.is_trained = False
            return {"status": "insufficient_data", "samples": len(expenses_df)}
            
        try:
            # Prepare training features
            X_train = []
            for idx, expense in expenses_df.iterrows():
                features = self.prepare_features(expenses_df, expense.to_dict())
                X_train.append(features.flatten())
            
            X_train = np.array(X_train)
            
            # Remove any NaN values and handle infinities
            X_train = np.nan_to_num(X_train)
            X_train = np.clip(X_train, -1e6, 1e6)  # Clip extreme values
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train Isolation Forest
            self.isolation_forest.fit(X_train_scaled)
            self.is_trained = True
            
            # Evaluate model (using training data as proxy)
            predictions = self.isolation_forest.predict(X_train_scaled)
            anomaly_indices = predictions == -1
            
            # Save model
            model_data = {
                'model': self.isolation_forest,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_size': len(X_train)
            }
            joblib.dump(model_data, 'models/isolation_forest_model.pkl')
            
            logger.info(f"ML model trained successfully on {len(X_train)} samples")
            
            return {
                "status": "success",
                "training_samples": len(X_train),
                "anomalies_detected": np.sum(anomaly_indices),
                "anomaly_rate": np.sum(anomaly_indices) / len(X_train),
                "feature_count": len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            self.is_trained = False
            return {"status": "error", "error": str(e)}
    
    def predict_anomaly(self, expenses_df: pd.DataFrame, current_expense: Dict) -> Dict[str, Any]:
        """Predict if current expense is anomalous using ML"""
        if not self.is_trained or len(expenses_df) < 5:
            return self.fallback_rule_based_detection(expenses_df, current_expense)
            
        try:
            # Prepare features for current expense
            features = self.prepare_features(expenses_df, current_expense)
            features = np.nan_to_num(features)
            features = np.clip(features, -1e6, 1e6)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomaly
            prediction = self.isolation_forest.predict(features_scaled)
            anomaly_score = self.isolation_forest.decision_function(features_scaled)
            
            # Convert to probability-like score (0-1, where 1 is more anomalous)
            confidence = (1 - (anomaly_score + 0.5)) * 2
            confidence = np.clip(confidence, 0, 1)[0]
            
            is_anomalous = prediction[0] == -1
            
            # Feature importance (simplified)
            feature_importance = self._get_feature_importance(features.flatten())
            
            return {
                "is_anomalous": bool(is_anomalous),
                "confidence_score": float(confidence),
                "anomaly_type": "ml_detected",
                "features_used": len(features[0]),
                "feature_importance": feature_importance,
                "raw_score": float(anomaly_score[0])
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self.fallback_rule_based_detection(expenses_df, current_expense)
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get simplified feature importance based on deviation from mean"""
        if not hasattr(self, 'feature_means') or len(self.feature_means) != len(features):
            return {}
            
        importance = {}
        for i, (feature, mean_val) in enumerate(zip(features, self.feature_means)):
            if self.feature_stds[i] > 0:
                z_score = abs(feature - mean_val) / self.feature_stds[i]
                importance[self.feature_names[i]] = float(min(z_score, 5.0))  # Cap at 5
        return importance
    
    def fallback_rule_based_detection(self, expenses_df: pd.DataFrame, current_expense: Dict) -> Dict[str, Any]:
        """Fallback to rule-based detection when ML fails"""
        anomalies = []
        confidence = 1.0
        
        # Rule 1: Amount significantly higher than historical average
        employee_history = expenses_df[expenses_df['employee_id'] == current_expense['employee_id']]
        if len(employee_history) > 0:
            avg_amount = employee_history['amount'].mean()
            std_amount = employee_history['amount'].std() if len(employee_history) > 1 else avg_amount * 0.5
            
            if current_expense['amount'] > avg_amount + 3 * std_amount:
                anomalies.append(f"Amount 3σ above personal average ({avg_amount:.2f} ± {std_amount:.2f})")
                confidence *= 0.3
        
        # Rule 2: Multiple similar recent submissions
        similar_expenses = employee_history[
            (employee_history['purpose'] == current_expense['purpose']) &
            (abs(employee_history['amount'] - current_expense['amount']) / current_expense['amount'] < 0.2)
        ]
        if len(similar_expenses) > 2:
            anomalies.append(f"Multiple similar {current_expense['purpose']} expenses")
            confidence *= 0.4
            
        # Rule 3: Department outlier using IQR
        dept_expenses = expenses_df[expenses_df['department'] == current_expense['department']]
        if len(dept_expenses) > 10:
            Q1 = dept_expenses['amount'].quantile(0.25)
            Q3 = dept_expenses['amount'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            if current_expense['amount'] > upper_bound:
                anomalies.append(f"Department IQR outlier (above {upper_bound:.2f})")
                confidence *= 0.5
        
        # Rule 4: Unusual purpose-amount combinations
        suspicious_patterns = {
            'Meals': 300,
            'Supplies': 1000,
            'Client Entertainment': 1500,
            'Travel': 2000
        }
        threshold = suspicious_patterns.get(current_expense['purpose'], float('inf'))
        if current_expense['amount'] > threshold:
            anomalies.append(f"Unusually high for {current_expense['purpose']} (threshold: {threshold})")
            confidence *= 0.6
        
        return {
            "is_anomalous": len(anomalies) > 0,
            "confidence_score": 1 - confidence,
            "anomaly_type": "rule_based",
            "detected_rules": anomalies,
            "features_used": 4
        }

class AdvancedRuleDetector:
    def __init__(self):
        self.suspicious_patterns = [
            {"purpose": "Client Entertainment", "amount_threshold": 500, "risk": "high"},
            {"purpose": "Travel", "amount_threshold": 1000, "risk": "medium"},
            {"purpose": "Meals", "amount_threshold": 200, "risk": "low"},
            {"purpose": "Supplies", "amount_threshold": 800, "risk": "medium"}
        ]
        
        self.risk_weights = {'high': 0.3, 'medium': 0.6, 'low': 0.8}
    
    def detect(self, expenses_df: pd.DataFrame, current_expense: Dict) -> Dict[str, Any]:
        anomalies = []
        confidence = 1.0
        
        # Pattern-based detection
        for pattern in self.suspicious_patterns:
            if (current_expense['purpose'] == pattern['purpose'] and 
                current_expense['amount'] > pattern['amount_threshold']):
                anomalies.append(
                    f"High amount for {pattern['purpose']} "
                    f"(threshold: {pattern['amount_threshold']})"
                )
                confidence *= self.risk_weights[pattern['risk']]
        
        # Rapid submission detection (using index as time proxy)
        employee_history = expenses_df[expenses_df['employee_id'] == current_expense['employee_id']]
        if len(employee_history) > 3:
            recent_submissions = len(employee_history.tail(3))
            if recent_submissions >= 3 and current_expense['amount'] > 100:
                anomalies.append("Rapid submission pattern detected")
                confidence *= 0.5
        
        # Department budget strain detection
        dept_expenses = expenses_df[expenses_df['department'] == current_expense['department']]
        if len(dept_expenses) > 0:
            total_dept_spend = dept_expenses['amount'].sum()
            avg_expense = dept_expenses['amount'].mean()
            if current_expense['amount'] > avg_expense * 2 and total_dept_spend > 10000:
                anomalies.append("High amount in high-spending department")
                confidence *= 0.7
        
        return {
            "is_anomalous": len(anomalies) > 0,
            "confidence_score": 1 - confidence,
            "detected_patterns": anomalies,
            "rule_count": len(anomalies)
        }

class EnsembleAnomalyDetector:
    def __init__(self):
        self.ml_detector = MLAnomalyDetector()
        self.rule_detector = AdvancedRuleDetector()
        self.ensemble_weights = {'ml': 0.7, 'rules': 0.3}
        
    def detect(self, expenses_df: pd.DataFrame, current_expense: Dict) -> Dict[str, Any]:
        """Ensemble detection combining ML and rules"""
        # ML detection
        ml_result = self.ml_detector.predict_anomaly(expenses_df, current_expense)
        
        # Rule-based detection
        rule_result = self.rule_detector.detect(expenses_df, current_expense)
        
        # Combine results using weighted average
        ml_confidence = ml_result["confidence_score"]
        rule_confidence = rule_result["confidence_score"]
        
        combined_confidence = (
            self.ensemble_weights['ml'] * ml_confidence + 
            self.ensemble_weights['rules'] * rule_confidence
        )
        
        # Final decision (either method flags it as anomalous with sufficient confidence)
        is_anomalous = (
            (ml_result["is_anomalous"] and ml_confidence > 0.3) or 
            (rule_result["is_anomalous"] and rule_confidence > 0.4)
        )
        
        # Determine anomaly type
        if ml_result["is_anomalous"] and rule_result["is_anomalous"]:
            anomaly_type = "both_ml_and_rules"
        elif ml_result["is_anomalous"]:
            anomaly_type = "ml_only"
        elif rule_result["is_anomalous"]:
            anomaly_type = "rules_only"
        else:
            anomaly_type = "normal"
        
        return {
            "is_anomalous": is_anomalous,
            "confidence_score": combined_confidence,
            "anomaly_type": anomaly_type,
            "ml_result": ml_result,
            "rule_result": rule_result,
            "final_decision": "anomalous" if is_anomalous else "normal",
            "ensemble_weights": self.ensemble_weights
        }
    
    def train_ml_model(self, expenses_df: pd.DataFrame) -> Dict[str, Any]:
        """Train the ML component of the ensemble"""
        return self.ml_detector.train_model(expenses_df)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the ensemble detector"""
        return {
            "detector_type": "Ensemble (Isolation Forest + Rule-Based)",
            "ml_trained": self.ml_detector.is_trained,
            "ensemble_weights": self.ensemble_weights,
            "rule_patterns_count": len(self.rule_detector.suspicious_patterns)

        }
