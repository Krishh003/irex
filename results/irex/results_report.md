# IREX implementation — results

- Input: 104 respondents, 102 binary questions, classes {'0': 39, '1': 35, '2': 30}.
- Exact duplicate responses: 3; grouped between training and test.
- First-fold baseline ANN: accuracy 0.2000, macro F1 0.1525, 20 test respondents.
- First-fold expert-direction **candidate** PAIs: 31; low-impact items: 40; all candidate removals are unapproved.
- First-fold exploratory screened ANN: accuracy 0.2000, 31 inputs excluded (not expert approved).
- Five-fold simple majority-class benchmark: mean accuracy 0.3178.
- Five-fold group-disjoint baseline: mean accuracy 0.3174, macro F1 0.2898.
- Five-fold fold-local screen: mean accuracy 0.2873, macro F1 0.2459.
- Mean screened-minus-baseline accuracy: -0.0300; fold-wise differences: [0.0, -0.04545, 0.14286, -0.04762, -0.2].
- Top 10 expert-direction mismatch question numbers: [83, 94, 75, 29, 89, 95, 24, 90, 21, 54].
- Top 10 approximate SHAP question numbers: [69, 56, 18, 77, 64, 45, 89, 54, 90, 40].

## Interpretation

Class 2 was **assumed** to be high-risk (not confirmed by documentation). The original notebook's use of *mean centered ALE* for directional anomalies was incorrect; it is replaced by the 1-minus-0 signed ALE effect. Sample size is small compared with 102 predictors. The questionnaire includes mental-health-related items, and model predictions should never be treated as clinical assessments. The screened model does not exceed the simple majority-class benchmark; do not claim it predicts reliably. No domain-expert selections were provided, and automatic feature removal is **exploratory**, not an expert-validated IREX result. Local explanations use a LIME-style weighted Ridge surrogate. SHAP is estimated by sampled permutation paths with training-fold references.
