# IREX implementation — results

- Input: 104 respondents, 102 binary questions, classes {'0': 39, '1': 35, '2': 30}.
- Exact duplicate responses: 3; grouped between training and test.
- First-fold baseline ANN: accuracy 0.3333, macro F1 0.2583, 21 test respondents.
- First-fold expert-direction **candidate** PAIs: 25; low-impact items: 48; all candidate removals are unapproved.
- First-fold exploratory screened ANN: accuracy 0.3810, 25 inputs excluded (not expert approved).
- Five-fold simple majority-class benchmark: mean accuracy 0.3748.
- Five-fold group-disjoint baseline: mean accuracy 0.2890, macro F1 0.2381.
- Five-fold fold-local screen: mean accuracy 0.3648, macro F1 0.3194.
- Mean screened-minus-baseline accuracy: +0.0757; fold-wise differences: [0.04762, 0.04762, 0.19048, 0.14286, -0.05].
- Top 10 expert-direction mismatch question numbers: [83, 79, 94, 93, 13, 74, 73, 75, 29, 52].
- Top 10 approximate SHAP question numbers: [69, 18, 87, 42, 76, 52, 5, 84, 85, 56].

## Interpretation

Class 2 was **assumed** to be high-risk (not confirmed by documentation). The original notebook's use of *mean centered ALE* for directional anomalies was incorrect; it is replaced by the 1-minus-0 signed ALE effect. Sample size is small compared with 102 predictors. The questionnaire includes mental-health-related items, and model predictions should never be treated as clinical assessments. The screened model does not exceed the simple majority-class benchmark; do not claim it predicts reliably. No domain-expert selections were provided, and automatic feature removal is **exploratory**, not an expert-validated IREX result. Local surrogates use a LIME-style algorithm but not the official `lime` library, which was unavailable in this environment.
