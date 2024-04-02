import papermill as pm
from pathlib import Path

for nb in Path('.').glob('*.ipynb'):
    pm.execute_notebook(
        input_path=nb,
        output_path=nb  # Path to save executed notebook
    )