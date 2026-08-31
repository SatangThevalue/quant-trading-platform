import sys
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from onnxmltools import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def export_model_to_onnx(model, feature_names, symbol="EURUSD", version="1.0"):
    """
    Phase 8: ONNX Deployment Pipeline
    Exports the trained LightGBM model and required configuration files for MT5.
    """
    logger.info(f"Starting ONNX export process for {symbol} (v{version})...")
    
    # 1. Prepare export directory
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "production")
    os.makedirs(export_dir, exist_ok=True)
    
    onnx_path = os.path.join(export_dir, f"{symbol}_model_v{version}.onnx")
    feature_order_path = os.path.join(export_dir, f"{symbol}_feature_order_v{version}.json")
    model_config_path = os.path.join(export_dir, f"{symbol}_model_config_v{version}.json")
    
    # 2. Convert LightGBM to ONNX
    try:
        # Define the input type: batch_size = None (dynamic), num_features = len(feature_names)
        initial_types = [('input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=12)
        
        # Save ONNX model
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Successfully exported ONNX model to {onnx_path}")
        
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        return False
        
    # 3. Export feature_order.json (Critical for MT5 Mismatch Prevention)
    feature_dict = {
        "feature_version": f"FS_V{version}",
        "symbol": symbol,
        "features": feature_names
    }
    with open(feature_order_path, "w") as f:
        json.dump(feature_dict, f, indent=4)
    logger.info(f"Exported feature order to {feature_order_path}")
        
    # 4. Export model_config.json (Thresholds and Risk Parameters)
    config_dict = {
        "model_version": f"V{version}",
        "symbol": symbol,
        "buy_threshold": 0.75,
        "sell_threshold": 0.25,
        "risk_percent": 1.0,
        "atr_sl": 1.5,
        "atr_tp": 3.0,
        "export_date": pd.Timestamp.now().isoformat()
    }
    with open(model_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Exported model config to {model_config_path}")
    
    return True

def validate_onnx_model(onnx_path, python_model, sample_features):
    """
    Validate that ONNX predictions exactly match Python predictions.
    """
    logger.info("Validating ONNX model against Python model...")
    
    # Python prediction
    py_preds = python_model.predict_proba(sample_features)[:, 1] # Get probability of class 1 (Buy)
    
    # ONNX prediction
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[1].name # Probabilities are usually in the second output
    
    # Convert to float32 as expected by ONNX
    onnx_inputs = {input_name: sample_features.astype(np.float32).values}
    onnx_preds_raw = sess.run([label_name], onnx_inputs)[0]
    
    # Extract probability for class 1
    onnx_preds = np.array([p[1] for p in onnx_preds_raw])
    
    # Calculate difference
    diff = np.abs(py_preds - onnx_preds).max()
    logger.info(f"Maximum prediction difference: {diff}")
    
    if diff < 1e-4:
        logger.info("✅ ONNX Validation PASSED: Predictions match Python exactly.")
        return True
    else:
        logger.error(f"❌ ONNX Validation FAILED: Max difference {diff} exceeds threshold.")
        return False

if __name__ == "__main__":
    logger.info("ONNX Export Pipeline (Dry-run mode, skipping execution as per instructions)")
