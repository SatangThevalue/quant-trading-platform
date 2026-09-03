import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

class MetaLabelingPipeline:
    def __init__(self, base_params, meta_params):
        self.base_model = lgb.LGBMClassifier(**base_params)
        self.meta_model = lgb.LGBMClassifier(**meta_params)
        
    def fit_predict(self, X_train, y_train, X_test, returns_test, cost):
        # 1. Train Base Model
        self.base_model.fit(X_train, y_train)
        
        # 2. Base Model Predictions on Train (OOB via TimeSeriesSplit to avoid data leak)
        tscv = TimeSeriesSplit(n_splits=3)
        meta_X = []
        meta_y = []
        
        # Fast simplistic OOB for meta labels
        split_idx = int(len(X_train) * 0.7)
        X_base_tr, y_base_tr = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
        X_base_val, y_base_val = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
        
        temp_model = lgb.LGBMClassifier(**self.base_model.get_params())
        temp_model.fit(X_base_tr, y_base_tr)
        
        base_preds_val = temp_model.predict(X_base_val)
        
        # Meta Label: 1 if Base Model is correct, 0 if wrong
        y_meta_val = (base_preds_val == y_base_val).astype(int)
        
        # Train Meta Model on the validation set of the base model
        self.meta_model.fit(X_base_val, y_meta_val)
        
        # 3. Final Prediction on Test Set
        base_prob = self.base_model.predict_proba(X_test)[:, 1]
        base_pred = (base_prob > 0.55).astype(int) # Default 1, else 0. We'll map to 1/-1 later
        
        meta_prob = self.meta_model.predict_proba(X_test)[:, 1]
        
        return base_prob, meta_prob
