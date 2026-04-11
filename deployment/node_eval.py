import os
import sys
import time
import json
import torch
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add /app to path for modular imports
sys.path.append("/app")

from utils.eval_utils import evaluate_model
from utils.data_loader import get_test_loader


TECHNIQUE_LABELS = {
    "B0": "Baseline",
    "Q1": "Quantification dynamique",
    "Q2": "Quantification statique PTQ",
    "Q3": "Quantification QAT",
    "Q4": "Poids FP16",
    "Q5": "Precision mixte",
    "P1": "Pruning non structure",
    "P2": "Pruning structure",
    "P3": "Pruning magnitude globale",
}

# MIT-BIH class labels (default). Can be overridden via CLASS_LABELS env var.
DEFAULT_CLASS_LABELS = ["N", "S", "V", "F", "Q"]


def _get_class_labels():
    raw = os.getenv("CLASS_LABELS", "")
    if raw.strip():
        labels = [x.strip() for x in raw.split(",") if x.strip()]
        return labels if labels else DEFAULT_CLASS_LABELS
    return DEFAULT_CLASS_LABELS


def _prediction_label(pred_idx, labels):
    if 0 <= pred_idx < len(labels):
        return labels[pred_idx]
    return str(pred_idx)


def _patient_id(specimen_id):
    year = datetime.now(timezone.utc).year
    return f"P-{year}-{specimen_id:03d}"

def measure_resources(interval=0.1):
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=None)
    memory_mb = process.memory_info().rss / (1024 * 1024)
    return cpu_percent, memory_mb

def main():
    model_path = os.getenv("MODEL_PATH")
    tech_id = os.getenv("TECH_ID")
    tech_name = os.getenv("TECH_NAME")
    vm_id = os.getenv("VM_ID", "UNKNOWN")
    data_path = os.getenv("DATA_PATH", "/mitbih/mitbih_test.csv")
    class_labels = _get_class_labels()
    
    if not model_path or not tech_id:
        print("MODEL_PATH and TECH_ID must be set.")
        return

    # Load data
    print(f"[{vm_id}] Loading test data from {data_path}...")
    test_df = pd.read_csv(data_path, header=None)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    x_test_t = torch.tensor(X_test).unsqueeze(1)
    y_test_t = torch.tensor(y_test)
    
    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=1, shuffle=False)

    # Load model
    print(f"[{vm_id}] Loading model {model_path}...")
    # Set quantization backend for Linux ARM64 (Docker on Mac)
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'

    import models.ecg_net as ecg_models
    # Assign architectures to top-level module to help torch.load
    sys.modules['__main__'].ECGNet1D = ecg_models.ECGNet1D
    sys.modules['__main__'].ECGNet1D_Narrow = ecg_models.ECGNet1D_Narrow
    sys.modules['__main__'].QuantizableClassifier = ecg_models.QuantizableClassifier
    sys.modules['__main__'].FP16Wrapper = ecg_models.FP16Wrapper
    sys.modules['__main__'].ManualMixedPrecisionWrapper = ecg_models.ManualMixedPrecisionWrapper

    loaded = None
    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[{vm_id}] Warning: Could not load {model_path} directly ({e}).")
        if tech_id in ["Q1", "Q2", "Q3"]:
            print(f"[{vm_id}] Re-applying {tech_id} transform to baseline...")
            baseline_path = "/app/results/baseline/baseline_best.pt"
            
            # Register safe globals for torch.load
            torch.serialization.add_safe_globals([ecg_models.ECGNet1D, ecg_models.ECGNet1D_Narrow, 
                                               ecg_models.QuantizableClassifier, ecg_models.FP16Wrapper, 
                                               ecg_models.ManualMixedPrecisionWrapper])
            
            model = ecg_models.ECGNet1D()
            loaded_baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
            if isinstance(loaded_baseline, torch.nn.Module):
                model.load_state_dict(loaded_baseline.state_dict())
            else:
                model.load_state_dict(loaded_baseline)
            
            if tech_id == "Q1":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            elif tech_id in ["Q2", "Q3"]:
                # For Q2/Q3 we need QuantizableClassifier wrapper
                model = ecg_models.QuantizableClassifier(model)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                # Calibrate slightly with a few samples
                with torch.no_grad():
                    for bx, _ in test_loader:
                        model(bx)
                        break
                torch.quantization.convert(model, inplace=True)
            loaded = model
        else:
            raise e

    if isinstance(loaded, torch.nn.Module):
        model = loaded
        # Post-load type enforcement for wrappers
        if tech_id == "Q4" and hasattr(model, "model"):
            model.model.half()
        elif tech_id == "Q5" and hasattr(model, "features"):
            model.features.float()
            model.classifier.half()
    elif isinstance(loaded, dict):
        print(f"[{vm_id}] Loaded a state dict. Rebuilding baseline architecture...")
        model = ecg_models.ECGNet1D()
        # Convert sparse tensors back to dense if necessary
        clean_state = {k: (v.to_dense() if hasattr(v, 'is_sparse') and v.is_sparse else v) for k, v in loaded.items()}
        model.load_state_dict(clean_state)
    else:
        raise ValueError(f"Unknown model type: {type(loaded)}")

    model.eval()

    # 1. Measure Latency (10 inferences)
    print(f"[{vm_id}] Running 10 inferences for latency measurement...")
    latencies = []
    # Warmup
    with torch.no_grad():
        for bx, _ in test_loader:
            _ = model(bx)
            break

    # Measurement loop
    process = psutil.Process(os.getpid())
    cpu_usages = []
    mem_usages = []
    
    for i, (bx, _) in enumerate(test_loader):
        if i >= 10: break
        
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(bx)
        dt = (time.perf_counter() - t0) * 1000 # ms
        latencies.append(dt)
        
        # Sample CPU/RAM (normalize by core count for true 0-100%)
        num_cpus = os.cpu_count() or 1
        cpu_usages.append(process.cpu_percent() / num_cpus)
        mem_usages.append(process.memory_info().rss / (1024 * 1024))

    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    avg_cpu = np.mean(cpu_usages)
    avg_mem = np.mean(mem_usages)

    # Measure Accuracy
    acc = 0.0
    if os.getenv("COLLECTIVE_MODE", "false").lower() != "true":
        print(f"[{vm_id}] Evaluating accuracy on full test set...")
        full_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=1024, shuffle=False)
        test_preds = []
        with torch.no_grad():
            for bx, _ in full_loader:
                logits = model(bx)
                test_preds.extend(logits.argmax(1).numpy())
                
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, test_preds)

    # Collective Mode: Process specific samples and save to shared volume
    collective_mode = os.getenv("COLLECTIVE_MODE", "false").lower() == "true"
    if collective_mode:
        num_samples = int(os.getenv("NUM_SAMPLES", "10"))
        save_dir = os.getenv("SHARED_DIR", "/app/results/phase5")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[{vm_id}] COLLECTIVE MODE: Processing {num_samples} samples...")
        
        count = 0
        with torch.no_grad():
            for bx, by in test_loader:
                for idx in range(len(bx)):
                    if count >= num_samples: break
                    sample = bx[idx:idx+1]

                    t0 = time.perf_counter()
                    outputs = model(sample)
                    inference_ms = (time.perf_counter() - t0) * 1000.0
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    pred_idx = int(pred.item())
                    pred_label = _prediction_label(pred_idx, class_labels)
                    technique_label = TECHNIQUE_LABELS.get(tech_id, tech_name or tech_id)
                    timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    num_cpus = os.cpu_count() or 1
                    process = psutil.Process(os.getpid())
                    cpu_usage_pct = round(process.cpu_percent(interval=None) / num_cpus, 2)
                    ram_usage_mb = round(process.memory_info().rss / (1024 * 1024), 2)
                    ram_usage_pct = round(psutil.virtual_memory().percent, 2)
                    res = {
                        "vm_id": vm_id,
                        "timestamp": timestamp_iso,
                        "technique": technique_label,
                        "prediction": pred_label,
                        "prediction_class": pred_idx,
                        "confidence": float(conf.item()),
                        "inference_time_ms": float(inference_ms),
                        "cpu_usage_pct": cpu_usage_pct,
                        "ram_usage_mb": ram_usage_mb,
                        "patient_id": _patient_id(count),
                        # Backward compatibility with existing aggregator/report code.
                        "accuracy": 1.0 if pred_idx == by[idx].item() else 0.0,
                        "cpu_percent": cpu_usage_pct,
                        "ram_percent": ram_usage_pct,
                        "specimen_id": count,
                        "tech_id": tech_id
                    }
                    
                    # 1. Save to shared volume
                    with open(os.path.join(save_dir, f"specimen_{count}_{vm_id}_pred.json"), 'w') as f:
                        json.dump(res, f)
                    
                    # 2. MQTT Telemetry (Supervision Phase 6)
                    mqtt_host = os.getenv("MQTT_HOST")
                    mqtt_token = os.getenv("MQTT_TOKEN")
                    if mqtt_host and mqtt_token:
                        try:
                            import paho.mqtt.client as mqtt
                            client = mqtt.Client()
                            client.username_pw_set(mqtt_token)
                            client.connect(mqtt_host, 1883, 60)
                            
                            payload = {
                                "vm_id": vm_id,
                                "timestamp": timestamp_iso,
                                "technique": technique_label,
                                "prediction": pred_label,
                                "prediction_class": pred_idx,
                                "confidence": float(conf.item()),
                                "inference_time_ms": float(inference_ms),
                                "cpu_usage_pct": cpu_usage_pct,
                                "ram_usage_mb": ram_usage_mb,
                                "patient_id": res["patient_id"],
                                # Keep legacy keys to avoid breaking dashboards/rules.
                                "cpu_percent": res["cpu_percent"],
                                "ram_percent": res["ram_percent"],
                                "specimen_id": count,
                                "tech_id": tech_id
                            }
                            client.publish("v1/devices/me/telemetry", json.dumps(payload))
                            client.disconnect()
                        except Exception as e:
                            print(f"[{vm_id}] MQTT Error: {e}")

                    count += 1
                if count >= num_samples: break
        print(f"[{vm_id}] Done collective processing.")
        return

    # Normal Mode (Existing Phase 3 logic)
    results = {
        "vm_id": vm_id,
        "tech_id": tech_id,
        "tech_name": tech_name,
        "accuracy": float(acc),
        "avg_latency_ms": float(avg_lat),
        "std_latency_ms": float(std_lat),
        "avg_cpu_percent": float(avg_cpu),
        "avg_ram_mb": float(avg_mem),
        "model_path": model_path
    }

    # Save results
    save_file = f"/app/results/phase3/{vm_id}_{tech_id}_results.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[{vm_id}] Results saved to {save_file}")

if __name__ == "__main__":
    main()
