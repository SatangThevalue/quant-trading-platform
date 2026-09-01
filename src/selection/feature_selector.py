import sys
import os
import pandas as pd
import numpy as np
from loguru import logger
import shap
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif

def run_feature_selection_pipeline(X, y):
    """
    Phase 5: Feature Selection Pipeline (Correlation -> Mutual Info -> SHAP)
    """
    logger.info(f"Starting Feature Selection Pipeline. Initial features: {X.shape[1]}")
    
    # 1. Correlation Filter
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
    X_corr = X.drop(columns=drop_corr)
    logger.info(f"Dropped {len(drop_corr)} correlated features. Remaining: {X_corr.shape[1]}")
    
    # 2. Mutual Information Filter
    # Only pick top 50 if we have more than that
    if X_corr.shape[1] > 50:
        mi_scores = mutual_info_classif(X_corr.fillna(0), y)
        mi_df = pd.DataFrame({'feature': X_corr.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        top_features = mi_df.head(50)['feature'].tolist()
        X_mi = X_corr[top_features]
        logger.info(f"Kept top 50 features via Mutual Information.")
    else:
        X_mi = X_corr
        
    # 3. SHAP Analysis (LightGBM)
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_mi, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_mi)
    
    # SHAP importance is the mean absolute value for each feature
    importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X_mi.columns, 'shap_importance': importance})
    shap_df = shap_df.sort_values('shap_importance', ascending=False)
    
    # Keep top 30
    final_features = shap_df.head(30)['feature'].tolist()
    logger.info(f"Final feature set selected via SHAP: {len(final_features)} features.")
    
    return final_features
