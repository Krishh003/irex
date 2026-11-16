# IREX

Run from the repository root using Python 3.10 and PowerShell.

```powershell
python -m pip install -r requirements.txt
python irex.py
```

C. E. Sosa-Espadas and colleagues, SoftwareX 23 (2023), article 101420. [Original paper](https://doi.org/10.1016/j.softx.2023.101420) · [Authors' code](https://github.com/Manuel080800/IREX)

`irex.py` adapts the group's executed notebook to a local command. It uses duplicate-aware folds and training-fold binary ALE for candidate detection, then computes permutation importance, sampled SHAP, and LIME-style weighted surrogates. It compares a baseline ANN, a majority baseline, and exploratory automatic removal. The supplied class-2 interpretation is an assumption. The original IREX process calls for expert review before refinement.

For one expert-reviewed refinement iteration, use the resulting `results/irex/pai_candidates.csv` for review, then run `python irex.py --approved-items path/to/approved.csv`. The CSV must contain a `Question_Number` column listing the first-fold candidates the reviewer approved for removal. The run exports separate scores and a model for that iteration.

Place the authorized Group 30 questionnaire inputs (`dataset.csv` and `ea.csv`) in `data/irex` before running. Saved and rerun aggregate reports are available in `results/irex_saved` and `results/irex` respectively.
