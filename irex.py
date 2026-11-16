# Adapted from Group 30 IREX_Completed_Results.ipynb.
from pathlib import Path
import argparse
import json
import warnings
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
HIGH_RISK_LABEL = 2    # Tentative ordinal interpretation; verify against dataset documentation
ALE_THRESHOLD = 0.01  # Probability-point difference (1 percentage point)
N_SPLITS = 5


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'irex')
parser.add_argument('--approved-items', type=Path,
                    help='CSV of domain-expert approved first-fold removals, with a Question_Number column.')
args = parser.parse_args()
DATA_PATH, EXPERT_PATH = args.data / 'dataset.csv', args.data / 'ea.csv'
OUT = ROOT / 'results' / 'irex'
OUT.mkdir(parents=True, exist_ok=True)


def display(value):
    print(value)


data = pd.read_csv(DATA_PATH)
expert = pd.read_csv(EXPERT_PATH).sort_values('Number').reset_index(drop=True)
assert 'Target' in data and {'Number', 'Question', 'RE'}.issubset(expert), 'Missing required columns'
X = data.drop(columns='Target').apply(pd.to_numeric, errors='raise')
y = data['Target'].astype(int)
assert not X.isna().any().any() and not y.isna().any(), 'Missing values exist'
assert X.shape[1] == len(expert) == 102, 'Expect 102 features and matching expert rows'
assert expert['Number'].tolist() == list(range(1, 103)), 'Expert item numbering mismatch'
assert all(str(c).startswith(f'{i}.') for i,c in enumerate(X.columns, 1)), 'Column sequence differs from expert item numbers'
assert X.isin([0,1]).all().all(), 'Binary ALE requires 0/1 inputs'
assert expert['RE'].isin([0,1]).all(), 'Expert RE must be binary'
assert HIGH_RISK_LABEL in y.unique(), 'High risk label not in target classes'
feature_names = X.columns.tolist()
expert['Feature'] = feature_names
expert_re = expert['RE'].to_numpy(dtype=int)
classes = sorted(y.unique().tolist())
# Identical responses remain a single group during every train/test partition.
groups = pd.util.hash_pandas_object(X, index=False).astype(str).to_numpy()
print('Rows, features:', X.shape, '| class counts:', y.value_counts().sort_index().to_dict())
print('Exact duplicate rows:', int(data.duplicated().sum()), '| duplicate response patterns:', int(X.duplicated().sum()))
print('Expert RE counts:', expert.RE.value_counts().sort_index().to_dict())
print('No missing data:', not X.isna().any().any())

def build_ann():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ann', MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                              max_iter=1500, early_stopping=True, validation_fraction=0.15,
                              n_iter_no_change=30, random_state=RANDOM_STATE))
    ])

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = list(splitter.split(X, y, groups))
train_idx, test_idx = folds[0]
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
base_model = build_ann().fit(X_train, y_train)
y_pred = base_model.predict(X_test)
base_accuracy = accuracy_score(y_test, y_pred)
print('Group-safe test set:',len(test_idx),'| train set:',len(train_idx))
print('Baseline test accuracy:',round(base_accuracy, 4))
print('Baseline balanced accuracy:',round(balanced_accuracy_score(y_test,y_pred),4),
      '| macro F1:', round(f1_score(y_test,y_pred,average='macro'),4))
print(classification_report(y_test,y_pred,labels=classes,zero_division=0))
cm = confusion_matrix(y_test, y_pred, labels=classes)
fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues');
for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,
       yticklabels=classes,xlabel='Predicted',ylabel='Actual',title='Baseline confusion matrix')
fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=145)

def binary_ale_deltas(model, inputs, positive_label=HIGH_RISK_LABEL):
    arr = inputs.to_numpy(dtype=float)
    pos = list(model.classes_).index(positive_label)
    deltas = np.zeros(arr.shape[1], dtype=float)
    for j in range(arr.shape[1]):
        low, high = arr.copy(), arr.copy()
        low[:,j] = 0; high[:,j] = 1
        deltas[j] = (model.predict_proba(pd.DataFrame(high,columns=inputs.columns))[:,pos] - model.predict_proba(pd.DataFrame(low,columns=inputs.columns))[:,pos]).mean()
    return deltas

def screen_pai(model, training_features):
    changes = binary_ale_deltas(model,training_features)
    expected_signed = np.where(expert_re == 1, 1, -1)*changes
    categories = np.select([np.abs(changes)<ALE_THRESHOLD,
                            expected_signed<=-ALE_THRESHOLD,
                            expected_signed>=ALE_THRESHOLD],
                            ['Low impact','Potentially anomalous','Direction agrees'],
                            default='Uncertain direction')
    tab=expert[['Number','Question','RE','Feature']].rename(columns={'Number':'Question_Number','RE':'Expert_RE'}).copy()
    tab['ALE_Change_1_minus_0'] = changes
    tab['ALE_Effect_Range'] = np.abs(changes)
    tab['Expert_Aligned_Change'] = expected_signed
    tab['Status'] = categories
    return tab

pai_table = screen_pai(base_model, X_train)
pai_table.to_csv(OUT/'pai_candidates.csv',index=False)
print('Item counts by screening status:')
print(pai_table.Status.value_counts().to_string())
print('\nTop candidate items (by magnitude of expert contradiction):')
display(pai_table.query('Status == "Potentially anomalous"').sort_values('Expert_Aligned_Change').head(15)[
    ['Question_Number','Question','Expert_RE','ALE_Change_1_minus_0','Status']])
fig,ax=plt.subplots(figsize=(14,4));a=np.arange(1,103)
colors=np.where(pai_table.Status.eq('Potentially anomalous'),'#b24a45','#477aa0')
ax.bar(a,pai_table.ALE_Change_1_minus_0,color=colors,width=.9)
ax.axhline(0,color='black',lw=.8);ax.axhline(ALE_THRESHOLD,color='grey',ls=':',lw=.8)
ax.axhline(-ALE_THRESHOLD,color='grey',ls=':',lw=.8)
ax.set(xlabel='Question number',ylabel='ALE change: answer 1 − answer 0',
       title='Binary-item ALE direction: red = expert-direction contradiction')
fig.tight_layout();fig.savefig(OUT/'ale_item_directions.png',dpi=145)

perm = permutation_importance(base_model,X_test,y_test,scoring='accuracy',
                              n_repeats=20,random_state=RANDOM_STATE,n_jobs=1)
permutation_table=pd.DataFrame({'Question_Number':np.arange(1,103),'Feature':feature_names,
                                'Permutation_Importance_Mean':perm.importances_mean,
                                'Permutation_Importance_Std':perm.importances_std})
permutation_table.sort_values('Permutation_Importance_Mean',ascending=False,inplace=True)
permutation_table.to_csv(OUT/'permutation_importance.csv',index=False)
print('Top 12 permutation importances (held-out responses; estimates are noisy):')
display(permutation_table[['Question_Number','Permutation_Importance_Mean','Permutation_Importance_Std']].head(12))
fig,ax=plt.subplots(figsize=(9,5));t=permutation_table.head(15).iloc[::-1]
ax.barh(t.Question_Number.astype(str),t.Permutation_Importance_Mean,xerr=t.Permutation_Importance_Std)
ax.set(xlabel='Mean decrease in held-out accuracy',ylabel='Question number',title='Top permutation importances')
fig.tight_layout();fig.savefig(OUT/'permutation_importance.png',dpi=145)

# Estimate interventional Shapley values along random feature-order paths.
rng = np.random.default_rng(RANDOM_STATE)
background = X_train.to_numpy(dtype=float)
N_SHAP_ROWS = min(8, len(X_test))
X_shap = X_test.iloc[:N_SHAP_ROWS].to_numpy(dtype=float)
n_features = X_train.shape[1]
class_index = list(base_model.classes_).index(HIGH_RISK_LABEL)
class_shap = []
for instance in X_shap:
    orders = np.stack([rng.permutation(n_features) for _ in range(64)])
    references = background[rng.integers(len(background), size=len(orders))]
    paths = np.broadcast_to(references[:, None, :], (len(orders), n_features + 1, n_features)).copy()
    values = paths[:, 0, :].copy()
    for step, features in enumerate(orders.T, start=1):
        values[np.arange(len(orders)), features] = instance[features]
        paths[:, step, :] = values
    frame = pd.DataFrame(paths.reshape(-1, n_features), columns=feature_names)
    scores = base_model.predict_proba(frame)[:, class_index].reshape(len(orders), n_features + 1)
    increments = np.empty_like(orders, dtype=float)
    np.put_along_axis(increments, orders, np.diff(scores, axis=1), axis=1)
    contribution = increments.mean(axis=0)
    np.testing.assert_allclose(contribution.sum(), (scores[:, -1] - scores[:, 0]).mean(), atol=1e-10)
    class_shap.append(contribution)
shap_importance=np.mean(np.abs(class_shap),axis=0)
shap_table=pd.DataFrame({'Question_Number':np.arange(1,103),'Feature':feature_names,'SHAP_Mean_Abs':shap_importance})
shap_table.sort_values('SHAP_Mean_Abs',ascending=False,inplace=True)
shap_table.to_csv(OUT/'shap_importance.csv',index=False)
print('SHAP evaluated on',N_SHAP_ROWS,'held-out rows. Top 12 class',HIGH_RISK_LABEL,'features:')
display(shap_table[['Question_Number','SHAP_Mean_Abs']].head(12))
fig,ax=plt.subplots(figsize=(9,5));t=shap_table.head(15).iloc[::-1]
ax.barh(t.Question_Number.astype(str),t.SHAP_Mean_Abs)
ax.set(xlabel='Mean |SHAP value| (probability)',ylabel='Question number',title='Sampled permutation SHAP importance')
fig.tight_layout();fig.savefig(OUT/'shap_importance.png',dpi=145)

def weighted_local_surrogate(model, instance, row_idx, n_samples=1800):
    gen=np.random.default_rng(RANDOM_STATE+int(row_idx))
    x0=instance.astype(int).copy()
    flips=gen.random((n_samples,len(x0)))<0.16
    perturb=np.logical_xor(x0[None,:].astype(bool),flips).astype(float)
    perturb[0]=x0
    target=model.predict_proba(pd.DataFrame(perturb,columns=feature_names))[:,list(model.classes_).index(HIGH_RISK_LABEL)]
    hd=np.sum(perturb!=x0,axis=1)
    weights=np.exp(-hd/(len(x0)*.26))
    local=Ridge(alpha=1.0).fit(perturb,target,sample_weight=weights)
    fitted=local.predict(perturb)
    quality=r2_score(target,fitted,sample_weight=weights)
    # Contributions compared with the training mean of each feature.
    contributions=local.coef_*(x0-X_train.to_numpy().mean(axis=0))
    top=np.argsort(-np.abs(contributions))[:12]
    return pd.DataFrame({'Test_Row':row_idx,'Question_Number':top+1,
        'Item_Value':x0[top], 'Local_Contribution':contributions[top],
        'Local_Coefficient':local.coef_[top], 'Local_R2':quality})

local_tables=[weighted_local_surrogate(base_model, X_test.iloc[i].to_numpy(),i)
              for i in range(min(3,len(X_test)))]
local_table=pd.concat(local_tables,ignore_index=True)
local_table.to_csv(OUT/'lime_style_local_explanations.csv',index=False)
print('Local weighted surrogate R² by test row:')
print(local_table.groupby('Test_Row')['Local_R2'].first().round(3).to_string())
display(local_table.head(15))

def run_fold(train_ix,test_ix):
    tr,te=X.iloc[train_ix],X.iloc[test_ix]
    yr,yt=y.iloc[train_ix],y.iloc[test_ix]
    m0=build_ann().fit(tr,yr)
    candidates=screen_pai(m0,tr)
    proposed=candidates.loc[candidates.Status.eq('Potentially anomalous'),'Feature'].tolist()
    remaining=[n for n in feature_names if n not in proposed]
    if len(remaining)==0: raise ValueError('All features proposed for removal; abort ablation')
    m1=build_ann().fit(tr[remaining],yr)
    p0=m0.predict(te);p1=m1.predict(te[remaining])
    majority_label=yr.mode().iloc[0]
    majority_preds=np.repeat(majority_label,len(yt))
    return {'Majority_Accuracy':accuracy_score(yt,majority_preds),
            'Majority_Macro_F1':f1_score(yt,majority_preds,labels=classes,average='macro',zero_division=0),
            'Baseline_Accuracy':accuracy_score(yt,p0),
            'Screened_Accuracy':accuracy_score(yt,p1),
            'Baseline_Macro_F1':f1_score(yt,p0,labels=classes,average='macro',zero_division=0),
            'Screened_Macro_F1':f1_score(yt,p1,labels=classes,average='macro',zero_division=0),
            'PAI_Candidates_Removed':len(proposed), 'Features_Remaining':len(remaining),
            'Test_Rows':len(te), 'Model':m1,'Removed_Features':proposed,'Prediction':p1}

cv_rows=[]
for fold_no,(tr_ix,te_ix) in enumerate(folds,1):
    result=run_fold(tr_ix,te_ix)
    cv_rows.append({'Fold':fold_no,**{k:v for k,v in result.items() if k not in ('Model','Removed_Features','Prediction')}})
    if fold_no==1: first_refined=result
cv_table=pd.DataFrame(cv_rows)
cv_table['Accuracy_Difference']=cv_table['Screened_Accuracy']-cv_table['Baseline_Accuracy']
cv_table.to_csv(OUT/'fivefold_cv_comparison.csv',index=False)
print('All comparisons use group-disjoint held-out responses and fold-specific PAI discovery:')
display(cv_table.round(4))
print('Mean majority-class accuracy:',round(cv_table.Majority_Accuracy.mean(),4))
print('Mean accuracy baseline:',round(cv_table.Baseline_Accuracy.mean(),4),
      '| screened:',round(cv_table.Screened_Accuracy.mean(),4),
      '| difference:',round(cv_table.Accuracy_Difference.mean(),4))
print('Mean macro F1 baseline:',round(cv_table.Baseline_Macro_F1.mean(),4),
      '| screened:',round(cv_table.Screened_Macro_F1.mean(),4))
print('Candidates removed per fold:',cv_table.PAI_Candidates_Removed.tolist())

ax=cv_table.plot(x='Fold',y=['Majority_Accuracy','Baseline_Accuracy','Screened_Accuracy'],kind='bar',figsize=(8,4),rot=0)
ax.set(title='Group-disjoint 5-fold evaluation (exploratory removal)',ylabel='Test accuracy',ylim=(0,1))
ax.legend(['Majority-class benchmark','Baseline ANN','Auto-screened ANN'])
plt.tight_layout();plt.savefig(OUT/'cross_validation.png',dpi=145)

comparison=pd.DataFrame([
    {'Model':'Baseline ANN','Features_Removed':0,'Features_Remaining':102,
     'Accuracy':accuracy_score(y_test,y_pred),
     'Macro_F1':f1_score(y_test,y_pred,average='macro',zero_division=0)},
    {'Model':'Exploratory auto-screened ANN (not expert approved)',
     'Features_Removed':len(first_refined['Removed_Features']),
     'Features_Remaining':first_refined['Features_Remaining'],
     'Accuracy':first_refined['Screened_Accuracy'],
     'Macro_F1':first_refined['Screened_Macro_F1']}
])
comparison.to_csv(OUT/'baseline_vs_refined.csv',index=False)
first_removed=set(first_refined['Removed_Features'])
data.drop(columns=list(first_removed)).to_csv(OUT/'exploratory_screened_dataset.csv',index=False)
joblib.dump(base_model, OUT/'baseline_ann.joblib')
joblib.dump(first_refined['Model'],OUT/'exploratory_screened_ann.joblib')
pd.DataFrame({'Row_Index':np.r_[train_idx,test_idx],
              'Partition':['train']*len(train_idx)+['test']*len(test_idx)}).to_csv(OUT/'first_fold_split.csv',index=False)

joined=pai_table.merge(shap_table[['Question_Number','SHAP_Mean_Abs']],on='Question_Number').merge(
    permutation_table[['Question_Number','Permutation_Importance_Mean']],on='Question_Number')
joined.to_csv(OUT/'explanation_method_comparison.csv',index=False)
display(comparison.round(4))
print('Files exported to:',OUT)
print('Important: predicted class 2 is assumed to be the high-risk group; verify before interpretation.')

summary={
 'scikit_learn_version':version('scikit-learn'),
 'respondents':int(len(data)),'features':int(X.shape[1]),'class_counts':{str(k):int(v) for k,v in y.value_counts().sort_index().items()},
 'duplicate_input_rows':int(X.duplicated().sum()),'test_size':len(test_idx),'high_risk_label_assumed':HIGH_RISK_LABEL,
 'baseline_first_fold_accuracy':float(cv_table.Baseline_Accuracy.iloc[0]),
 'auto_screened_first_fold_accuracy':float(cv_table.Screened_Accuracy.iloc[0]),
 'auto_screened_first_fold_features_removed':int(cv_table.PAI_Candidates_Removed.iloc[0]),
 'pai_first_fold_count':int((pai_table.Status=='Potentially anomalous').sum()),
 'low_impact_first_fold_count':int((pai_table.Status=='Low impact').sum()),
 'fivefold_mean_majority_accuracy':float(cv_table.Majority_Accuracy.mean()),
 'fivefold_mean_baseline_accuracy':float(cv_table.Baseline_Accuracy.mean()),
 'fivefold_mean_auto_screened_accuracy':float(cv_table.Screened_Accuracy.mean()),
 'fivefold_mean_baseline_macro_f1':float(cv_table.Baseline_Macro_F1.mean()),
 'fivefold_mean_auto_screened_macro_f1':float(cv_table.Screened_Macro_F1.mean()),
 'fivefold_accuracy_differences':cv_table.Accuracy_Difference.round(5).tolist(),
 'top_pai_question_numbers':pai_table.query('Status == "Potentially anomalous"').sort_values('Expert_Aligned_Change').Question_Number.head(10).astype(int).tolist(),
 'top_shap_question_numbers':shap_table.Question_Number.head(10).astype(int).tolist(),
 'top_permutation_question_numbers':permutation_table.Question_Number.head(10).astype(int).tolist(),
 'expert_confirmation_provided':False,'local_explanation_method':'LIME-style weighted Ridge surrogate, NOT official lime',
 'experimental_note':'All screened comparisons are automatic ablations, not expert-confirmed IREX refinement. Small sample and high dimensionality limit inference.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
report=f"""# IREX implementation — results

- Input: {len(data)} respondents, {X.shape[1]} binary questions, classes {summary['class_counts']}.
- Exact duplicate responses: {summary['duplicate_input_rows']}; grouped between training and test.
- First-fold baseline ANN: accuracy {summary['baseline_first_fold_accuracy']:.4f}, macro F1 {cv_table.Baseline_Macro_F1.iloc[0]:.4f}, {len(test_idx)} test respondents.
- First-fold expert-direction **candidate** PAIs: {summary['pai_first_fold_count']}; low-impact items: {summary['low_impact_first_fold_count']}; all candidate removals are unapproved.
- First-fold exploratory screened ANN: accuracy {summary['auto_screened_first_fold_accuracy']:.4f}, {summary['auto_screened_first_fold_features_removed']} inputs excluded (not expert approved).
- Five-fold simple majority-class benchmark: mean accuracy {summary['fivefold_mean_majority_accuracy']:.4f}.
- Five-fold group-disjoint baseline: mean accuracy {summary['fivefold_mean_baseline_accuracy']:.4f}, macro F1 {summary['fivefold_mean_baseline_macro_f1']:.4f}.
- Five-fold fold-local screen: mean accuracy {summary['fivefold_mean_auto_screened_accuracy']:.4f}, macro F1 {summary['fivefold_mean_auto_screened_macro_f1']:.4f}.
- Mean screened-minus-baseline accuracy: {cv_table.Accuracy_Difference.mean():+.4f}; fold-wise differences: {summary['fivefold_accuracy_differences']}.
- Top 10 expert-direction mismatch question numbers: {summary['top_pai_question_numbers']}.
- Top 10 approximate SHAP question numbers: {summary['top_shap_question_numbers']}.

## Interpretation

Class 2 was **assumed** to be high-risk (not confirmed by documentation). The original notebook's use of *mean centered ALE* for directional anomalies was incorrect; it is replaced by the 1-minus-0 signed ALE effect. Sample size is small compared with 102 predictors. The questionnaire includes mental-health-related items, and model predictions should never be treated as clinical assessments. The screened model does not exceed the simple majority-class benchmark; do not claim it predicts reliably. No domain-expert selections were provided, and automatic feature removal is **exploratory**, not an expert-validated IREX result. Local explanations use a LIME-style weighted Ridge surrogate. SHAP is estimated by sampled permutation paths with training-fold references.
"""
(OUT/'results_report.md').write_text(report, encoding='utf-8')
print(report)

if args.approved_items:
    approved = set(pd.read_csv(args.approved_items)['Question_Number'].astype(int))
    candidates = pai_table.loc[pai_table.Status.eq('Potentially anomalous')]
    if not approved.issubset(set(candidates.Question_Number)):
        parser.error('Approved question numbers must come from first-fold PAI candidates.')
    removed = set(candidates.loc[candidates.Question_Number.isin(approved), 'Feature'])
    remaining = [name for name in feature_names if name not in removed]
    approved_model = build_ann().fit(X_train[remaining], y_train)
    predictions = approved_model.predict(X_test[remaining])
    approved_results = {
        'approved_question_numbers': sorted(approved),
        'test_accuracy': accuracy_score(y_test, predictions),
        'test_macro_f1': f1_score(y_test, predictions, labels=classes, average='macro', zero_division=0),
    }
    (OUT / 'expert_reviewed_first_fold.json').write_text(json.dumps(approved_results, indent=2), encoding='utf-8')
    joblib.dump(approved_model, OUT / 'expert_reviewed_first_fold.joblib')
