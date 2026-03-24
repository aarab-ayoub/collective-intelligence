from pathlib import Path

SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "mitbih"
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
OPTIMIZATION_DIR = RESULTS_DIR / "optimization"
GRAPHS_DIR = RESULTS_DIR / "graphs_and_images"

# Baseline model filename
BASELINE_MODEL_NAME = "baseline_best.pt"
BASELINE_MODEL_PATH = BASELINE_DIR / BASELINE_MODEL_NAME
