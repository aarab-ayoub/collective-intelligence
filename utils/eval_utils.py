import torch
import time
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .data_loader import get_test_loader

def evaluate_model(model, model_name, technique_id, technique_name, save_path=None, test_loader=None, device="cpu", save_model_path=None, sparse_state_dict=None):
    """
    Evaluates a model (accuracy, F1, size, inference time) and saves metrics to a JSON file.
    """
    if test_loader is None:
        test_loader = get_test_loader()
        
    model.eval()
    model.to(device)
    
    test_preds, test_targs = [], []
    
    # Warmup
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            _ = model(bx)
            break
            
    # Measure inference time over the whole test set
    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_targs.extend(by.numpy())
            
    inference_time = (time.perf_counter() - t0) * 1000  # ms
    avg_inference_ms = inference_time / len(test_targs)
    
    test_preds = np.array(test_preds)
    test_targs = np.array(test_targs)
    
    acc = accuracy_score(test_targs, test_preds)
    f1 = f1_score(test_targs, test_preds, average="macro", zero_division=0)
    
    # Measure size
    if save_model_path:
        state_to_save = sparse_state_dict if sparse_state_dict is not None else model
        torch.save(state_to_save, save_model_path)
        model_size_bytes = os.path.getsize(save_model_path)
    else:
        import io
        buffer = io.BytesIO()
        state_to_save = sparse_state_dict if sparse_state_dict is not None else model
        torch.save(state_to_save, buffer)
        model_size_bytes = buffer.getbuffer().nbytes
        
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    metrics = {
        "id": technique_id,
        "technique": technique_name,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "size_mb": float(model_size_mb),
        "inference_ms": float(avg_inference_ms)
    }
    
    print(f"--- {technique_name} ({technique_id}) ---")
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Macro F1    : {f1:.4f}")
    print(f"Size        : {model_size_mb:.3f} MB")
    print(f"Inv. Time   : {avg_inference_ms:.4f} ms/sample")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
    return metrics
