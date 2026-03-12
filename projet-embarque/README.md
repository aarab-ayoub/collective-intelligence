# Inférence Médicale Embarquée

**Module:** Systèmes embarqués et objets connectés — Master Data Science, ENS Martil  
**Encadrant:** Said Ohamouddou  
**Date:** 1er mars 2026  
**Équipe:** Groupes de 3 étudiants

---

## Objectifs

1. Entraîner un modèle deep learning de base sur un dataset médical public.
2. Appliquer **8 techniques d'optimisation** (quantification + élagage) pour réduire la taille et le coût d'inférence.
3. Déployer et évaluer les modèles sur **3 machines virtuelles** simulant du matériel de plus en plus contraint.
4. Combiner les prédictions des 3 VMs en un système d'**intelligence collective**.
5. Monitorer l'ensemble via une **plateforme IoT (ThingsBoard)** avec télémétrie MQTT.

---

## Structure du dépôt

```
projet-embarque/
├── README.md
├── environment/              # Infrastructure VM (Docker / Vagrant)
│   ├── docker-compose.yml
│   └── check_resources.sh
├── dataset/                  # Téléchargement & preprocessing
│   ├── download.sh
│   └── preprocess.py
├── baseline/                 # Modèle de base
│   ├── train.py
│   └── evaluate.py
├── optimization/             # 8 techniques d'optimisation
│   ├── Q1_dynamic_quant/
│   ├── Q2_static_ptq/
│   ├── Q3_qat/
│   ├── Q4_weight_only/
│   ├── Q5_mixed_precision/
│   ├── P1_unstructured/
│   ├── P2_structured/
│   └── P3_magnitude/
├── deployment/               # Scripts de déploiement & mesure sur VMs
│   ├── deploy.py
│   └── measure.py
├── collective/               # Orchestrateur & mécanismes de vote
│   └── orchestrator.py
├── thingsboard/              # Client MQTT & dashboards
│   ├── mqtt_client.py
│   └── dashboards/
├── results/                  # Tables CSV & figures
└── report/                   # Rapport final PDF
```

---

## Environnement virtuel

| VM  | CPU   | RAM    | Profil                    |
|-----|-------|--------|---------------------------|
| VM1 | 1 core| 500 MB | Capteur IoT bas de gamme  |
| VM2 | 2 cores| 1 GB  | Passerelle IoT             |
| VM3 | 2 cores| 2 GB  | Serveur edge léger         |

---

## Techniques d'optimisation

| ID | Technique                       |
|----|---------------------------------|
| Q1 | Quantification dynamique        |
| Q2 | PTQ statique                    |
| Q3 | Quantification QAT              |
| Q4 | Quantification poids seulement  |
| Q5 | Quantification mixte            |
| P1 | Élagage non structuré           |
| P2 | Élagage structuré               |
| P3 | Élagage par magnitude           |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <repo-url>
cd projet-embarque

# 2. Créer un environnement virtuel Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Lancer l'infrastructure Docker
cd environment
docker-compose up -d

# 4. Télécharger et préparer le dataset
cd ../dataset
bash download.sh
python preprocess.py

# 5. Entraîner le modèle de base
cd ../baseline
python train.py
```

---

## Résultats (Phase 3 — Matrice 3×8)

Voir `results/deployment_matrix.csv` après exécution de `deployment/measure.py`.

---

## Monitoring

ThingsBoard Community Edition est déployé via Docker.  
Le client MQTT (`thingsboard/mqtt_client.py`) publie les métriques après chaque inférence.
