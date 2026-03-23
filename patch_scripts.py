import glob
from pathlib import Path

# 1. Update utils/utils.py to save full model
utils_file = Path("utils/utils.py")
content = utils_file.read_text()
content = content.replace(
    "model.state_dict() if hasattr(model, 'state_dict') else model",
    "model"
)
utils_file.write_text(content)

# 2. Patch all optimize.py scripts
for optimize_script in glob.glob("optimization/*/optimize.py"):
    path = Path(optimize_script)
    content = path.read_text()
    
    # Imports
    content = content.replace("from utils import", "from utils.utils import")
    
    # Baseline paths (was baseline/weights/baseline_mitbih_csv.pt)
    content = content.replace(
        "baseline/weights/baseline_mitbih_csv.pt",
        "results/baseline/baseline_best.pt"
    )
    content = content.replace(
        "baseline_weights =",
        "baseline_model_path ="
    )
    
    # Loading full model instead of state dict
    content = content.replace(
        """model = ECGNet1D()
    model.load_state_dict(torch.load(baseline_model_path, map_location="cpu"))""",
        """model = torch.load(baseline_model_path, map_location="mps" if torch.backends.mps.is_available() else "cpu")"""
    )
    # Some had different variable names
    content = content.replace(
        """base_model = ECGNet1D()
    base_model.load_state_dict(torch.load(baseline_model_path, map_location="cpu"))""",
        """base_model = torch.load(baseline_model_path, map_location="mps" if torch.backends.mps.is_available() else "cpu")"""
    )
    
    # MPS device
    if not any([q in optimize_script for q in ["Q1", "Q2", "Q3"]]):
        content = content.replace('device="cpu"', 'device="mps" if torch.backends.mps.is_available() else "cpu"')
    
    path.write_text(content)
print("Patching complete!")
