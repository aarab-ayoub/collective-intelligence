import os
import time
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import smtplib
import ssl
import urllib.request
import urllib.error

# Configuration from Phase 4 results
PROJECT_ROOT = Path(__file__).resolve().parent.parent

NODES = {
    "VM1": {"weight": 0.9748, "results_file": "VM1_pred.json"},
    "VM2": {"weight": 0.9773, "results_file": "VM2_pred.json"},
    "VM3": {"weight": 0.9821, "results_file": "VM3_pred.json"}
}

SHARED_DIR = Path(os.getenv("SHARED_DIR", str(PROJECT_ROOT / "results" / "phase5")))
CONFIDENCE_THRESHOLD = 0.70


def _prediction_class(data):
    if "prediction_class" in data:
        return int(data["prediction_class"])
    return int(data.get("prediction", 0))


def _cpu_usage_pct(data):
    if "cpu_usage_pct" in data:
        return max(0.0, min(float(data["cpu_usage_pct"]), 100.0))
    return max(0.0, min(float(data.get("cpu_percent", 0)), 100.0))


def _ram_usage_pct(data):
    if "ram_percent" in data:
        return max(0.0, min(float(data["ram_percent"]), 100.0))
    ram_mb = float(data.get("ram_usage_mb", 0))
    ram_cap_mb = float(data.get("ram_capacity_mb", 0))
    if ram_cap_mb > 0:
        return max(0.0, min((ram_mb / ram_cap_mb) * 100.0, 100.0))
    return 0.0


def _scaled_int(value, factor=100):
    """Convert float telemetry to integer-friendly values for bar-chart widgets."""
    return int(round(float(value) * factor))


def _flatten_node_payload(node_data):
    """Flatten per-node telemetry so a single aggregator message can drive dashboards."""
    flat = {}
    for node_id, data in node_data.items():
        prefix = node_id.lower()
        confidence = float(data.get("confidence", 0.0))
        inference_time_ms = float(data.get("inference_time_ms", 0.0))
        cpu_usage_pct = _cpu_usage_pct(data)
        ram_usage_mb = float(data.get("ram_usage_mb", 0.0))
        ram_usage_pct = _ram_usage_pct(data)
        cpu_cores = int(data.get("cpu_cores", 0) or 0)
        ram_capacity_mb = float(data.get("ram_capacity_mb", 0.0))

        flat[f"{prefix}_prediction_class"] = _prediction_class(data)
        flat[f"{prefix}_confidence"] = confidence
        flat[f"{prefix}_inference_time_ms"] = inference_time_ms
        flat[f"{prefix}_cpu_usage_pct"] = cpu_usage_pct
        flat[f"{prefix}_ram_usage_mb"] = ram_usage_mb
        flat[f"{prefix}_ram_usage_pct"] = ram_usage_pct
        flat[f"{prefix}_cpu_cores"] = cpu_cores
        flat[f"{prefix}_ram_capacity_mb"] = ram_capacity_mb
        # Integer telemetry mirrors for widgets that do not handle floats reliably.
        flat[f"{prefix}_confidence_x100"] = _scaled_int(confidence)
        flat[f"{prefix}_inference_time_ms_x100"] = _scaled_int(inference_time_ms)
        flat[f"{prefix}_cpu_usage_pct_x100"] = _scaled_int(cpu_usage_pct)
        flat[f"{prefix}_ram_usage_mb_x100"] = _scaled_int(ram_usage_mb)
        flat[f"{prefix}_ram_usage_pct_x100"] = _scaled_int(ram_usage_pct)
        flat[f"{prefix}_ram_capacity_mb_x100"] = _scaled_int(ram_capacity_mb)
        flat[f"{prefix}_tech_id"] = str(data.get("tech_id", ""))
    return flat


def _load_prediction_file(file_path, min_mtime=None):
    if not os.path.exists(file_path):
        return None
    if min_mtime is not None and os.path.getmtime(file_path) < min_mtime:
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _send_alerts_via_email(alerts):
    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        print("[Aggregator] SMTP_HOST not set; skipping email alerts")
        return False
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    sender = os.getenv("ALERT_FROM", smtp_user)
    recipients = [r.strip() for r in os.getenv("ALERT_TO", "").split(",") if r.strip()]
    if not recipients:
        print("[Aggregator] ALERT_TO not set; skipping email alerts")
        return False

    subject = f"Collective Intelligence Alerts ({len(alerts)})"
    body_lines = [f"Alert summary ({len(alerts)}):"]
    for a in alerts:
        body_lines.append(f"- {a['node']}: CPU={a['cpu_pct']}%, RAM={a['ram_pct']}% | {a['msg']}")
    body = "\n".join(body_lines)

    message = f"From: {sender}\r\nTo: {', '.join(recipients)}\r\nSubject: {subject}\r\n\r\n{body}"

    if smtp_port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(sender, recipients, message)
    else:
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        try:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(sender, recipients, message)
        finally:
            server.quit()
    print(f"[Aggregator] Email alert sent to: {recipients}")
    return True


def _send_alerts_via_webhook(alerts):
    url = os.getenv("PAGER_WEBHOOK_URL")
    if not url:
        print("[Aggregator] PAGER_WEBHOOK_URL not set; skipping webhook alerts")
        return False
    payload = json.dumps({"alerts": alerts}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.getcode()
            print(f"[Aggregator] Webhook alert posted, status={status}")
            return True
    except urllib.error.URLError as e:
        print(f"[Aggregator] Webhook error: {e}")
        return False

def wait_for_predictions(specimen_id, run_start):
    """Wait for all nodes to output their prediction for a given specimen."""
    preds = {}
    start_time = time.time()
    timeout = 30 # seconds
    
    while len(preds) < 3 and (time.time() - start_time) < timeout:
        for node_id, info in NODES.items():
            if node_id not in preds:
                file_path = os.path.join(SHARED_DIR, f"specimen_{specimen_id}_{info['results_file']}")
                loaded = _load_prediction_file(file_path, min_mtime=run_start)
                if loaded is not None:
                    preds[node_id] = loaded
        time.sleep(0.1)

    if len(preds) < 3:
        for node_id, info in NODES.items():
            if node_id not in preds:
                file_path = os.path.join(SHARED_DIR, f"specimen_{specimen_id}_{info['results_file']}")
                loaded = _load_prediction_file(file_path)
                if loaded is not None:
                    preds[node_id] = loaded
        if preds:
            print(f"[Aggregator] Using existing specimen files for specimen {specimen_id} after waiting for fresh node output.")

    return preds

def solve_collective(specimen_id, run_start):
    print(f"\n[Aggregator] Solving Specimen {specimen_id}...")
    node_data = wait_for_predictions(specimen_id, run_start)
    
    if len(node_data) < 3:
        print(f"[Aggregator] Error: Timeout waiting for nodes for specimen {specimen_id}")
        return None

    # 1. Check Load Balancing / Resource Overload and collect alerts
    alerts = []
    for node_id, data in node_data.items():
        cpu_pct = _cpu_usage_pct(data)
        ram_pct = _ram_usage_pct(data)
        if cpu_pct > 85 or ram_pct > 90:
            msg = f"{node_id} overloaded - CPU: {cpu_pct}%, RAM: {ram_pct}%"
            print(f"[Aggregator] WARNING: {msg}")
            alerts.append({
                "node": node_id,
                "cpu_pct": cpu_pct,
                "ram_pct": ram_pct,
                "msg": msg,
            })
            # In a real system, the orchestrator would redirect. Here we just log.

    # 2. Weighted Voting & Confidence
    total_weight = sum(NODES[node_id]['weight'] for node_id in node_data)
    weighted_votes = {}
    
    for node_id, data in node_data.items():
        pred_class = _prediction_class(data)
        confidence = data['confidence']
        weight = NODES[node_id]['weight']
        
        # Vote contribution: normalized weight * confidence
        vote_score = (weight / total_weight) * confidence
        weighted_votes[pred_class] = weighted_votes.get(pred_class, 0) + vote_score

    final_class = max(weighted_votes, key=weighted_votes.get)
    final_confidence = weighted_votes[final_class]
    
    print(f"[Aggregator] Collective Decision: Class {final_class} with confidence {final_confidence:.4f}")

    # 3. Validation by Confidence
    validation_needed = False
    if final_confidence < CONFIDENCE_THRESHOLD:
        print(f"[Aggregator] Confidence {final_confidence:.4f} < {CONFIDENCE_THRESHOLD}. RE-TRIGGERING validation...")
        # Simulation: In a real system, we'd request more data. Here we just flag it.
        validation_needed = True

    # 4. MQTT Telemetry (Supervision Phase 6)
    mqtt_host = os.getenv("MQTT_HOST")
    mqtt_token = os.getenv("MQTT_TOKEN")
    if mqtt_host and mqtt_token:
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client()
            client.username_pw_set(mqtt_token)
            client.connect(mqtt_host, 1883, 60)
            client.loop_start()
            
            payload = {
                "collective_class": int(final_class),
                "collective_confidence": float(final_confidence),
                "collective_confidence_x100": _scaled_int(final_confidence),
                "consensus_achieved": len(set(_prediction_class(n) for n in node_data.values())) == 1,
                "revalidation_triggered": validation_needed,
                "specimen_id": specimen_id,
                # Full VM details for drill-down from a single telemetry stream.
                "nodes_payload": json.dumps(node_data),
            }
            payload.update(_flatten_node_payload(node_data))
            # Add alerts summary for dashboard
            payload["alert_count"] = len(alerts)
            payload["alerts"] = json.dumps(alerts)
            msg_info = client.publish("v1/devices/me/telemetry", json.dumps(payload), qos=1)
            msg_info.wait_for_publish(timeout=5)
            client.loop_stop()
            client.disconnect()
            print(f"[Aggregator] MQTT telemetry sent successfully")
        except Exception as e:
            print(f"[Aggregator] MQTT Error: {e}")

    # 5. Send external alerts (email / webhook) if any
    if alerts:
        try:
            _send_alerts_via_email(alerts)
        except Exception as e:
            print(f"[Aggregator] Email alert error: {e}")
        try:
            _send_alerts_via_webhook(alerts)
        except Exception as e:
            print(f"[Aggregator] Webhook alert error: {e}")

    return {
        "specimen_id": specimen_id,
        "final_class": int(final_class),
        "confidence": float(final_confidence),
        "validation_needed": validation_needed,
        "nodes": node_data
    }

def main():
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    run_start = time.time()

    stale_files = [path for path in SHARED_DIR.glob("specimen_*_pred.json") if path.stat().st_mtime < run_start]
    if stale_files:
        print(f"[Aggregator] Found {len(stale_files)} stale result file(s) from a previous run. They will be ignored.")
        
    results = []
    # We will evaluate 10 specimens as requested
    for i in range(10):
        res = solve_collective(i, run_start)
        if res:
            results.append(res)
            
    output_file = os.path.join(SHARED_DIR, "collective_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Summarize consensus
    if results:
        consensus_count = 0
        for r in results:
            unique_preds = set(_prediction_class(n) for n in r['nodes'].values())
            if len(unique_preds) == 1:
                consensus_count += 1
        
        print(f"\n[Aggregator] FINAL EVALUATION SUMMARY:")
        print(f"Total Examples: {len(results)}")
        print(f"Consensus Rate: {consensus_count/len(results)*100:.1f}%")
        print(f"Validation Re-triggers: {sum(1 for r in results if r['validation_needed'])}")

if __name__ == "__main__":
    main()
