import glob
from pathlib import Path
for optimize_script in glob.glob("optimization/*/optimize.py"):
    path = Path(optimize_script)
    content = path.read_text()
    content = content.replace("Path(__file__).resolve().parent.parent))", "Path(__file__).resolve().parent.parent.parent))")
    path.write_text(content)
print("Fixed sys.path")
