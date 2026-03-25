# ThingsBoard Supervision Hub

This directory contains the configuration and instructions for the Phase 6 Supervision system.

## Components
- `dashboard.json`: A pre-configured dashboard template for visualizing ECG predictions, node health (CPU/RAM), and collective consensus.
- `README.md`: This guide.

## Setup Instructions

### 1. Start the ThingsBoard Server
Ensure you have Docker installed and run:
```bash
docker compose -f deployment/docker-compose.yml up -d thingsboard
```
Wait for the service to be healthy at `http://localhost:8080`.

### 2. Provision Devices
In the ThingsBoard UI, create the following devices:
- **VM1**: Select "MQTT" and set Access Token to `VM1_TOKEN`.
- **VM2**: Set Access Token to `VM2_TOKEN`.
- **VM3**: Set Access Token to `VM3_TOKEN`.
- **Aggregator**: Set Access Token to `AGGREGATOR_TOKEN`.

### 3. Import Dashboard
1. Go to **Dashboards** -> **Import Dashboard**.
2. Select `dashboard.json`.
3. Map the aliases to the devices created in Step 2.

## Telemetry Format
The nodes report JSON telemetry to `v1/devices/me/telemetry` with the following fields:
```json
{
  "vm_id": "VM1",
  "technique": "Q2",
  "prediction": 0,
  "confidence": 0.99,
  "cpu_usage_pct": 12.5,
  "ram_usage_mb": 450,
  "specimen_id": 5
}
```
