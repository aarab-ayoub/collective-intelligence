import re
from pathlib import Path

# 1. Update baseline/train_mitbih_csv.py
bl = Path("baseline/train_mitbih_csv.py")
content = bl.read_text()

content = content.replace('results/mitbih_csv/baseline_best.pt', 'results/baseline/baseline_best.pt')
content = content.replace('results/mitbih_csv/baseline_metrics.json', 'results/baseline/baseline_metrics.json')
content = content.replace('results/mitbih_csv/baseline_plots.png', 'results/graphs_and_images/baseline_plots.png')
content = content.replace('torch.save(best_model_state, save_model_path)', 'torch.save(model, save_model_path)')
content = content.replace("device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))", 
                          "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
bl.write_text(content)

# 2. Update compare_techniques.py
ct = Path("optimization/compare_techniques.py")
content = ct.read_text()
content = content.replace('results/mitbih_csv/baseline_cnn_metrics.json', 'results/baseline/baseline_metrics.json')
content = content.replace('results/optimization/optimization_pareto_front.png', 'results/graphs_and_images/optimization_pareto_front.png')
ct.write_text(content)

print("Paths updated!")
