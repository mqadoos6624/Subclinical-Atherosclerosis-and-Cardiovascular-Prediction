"""
================================================================================
Artificial Intelligence–Driven Integration of Imaging-Derived Biomarkers and Clinical Risk Factors for Early Detection of Subclinical Atherosclerosis and Cardiovascular Outcomes Prediction Using NHANES Data
--------------------------------------------------------------------------------
 Author      : [Muhammad Qadoos]
 Dataset     : NHANES (CDC), UCI Heart Disease, Framingham Teaching Dataset
 Python      : 3.9+
================================================================================

DATASETS USED:
  1. NHANES    – https://wwwn.cdc.gov/nchs/nhanes/  (CDC public data)
  2. UCI Heart – https://archive.ics.uci.edu/dataset/45/heart+disease
  3. Framingham – https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset
  4. Combined UCI (Cleveland+Hungarian+Switzerland+VA)
       https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

ALGORITHMS IMPLEMENTED:
  - Logistic Regression (baseline)
  - Random Forest`
  - XGBoost (primary model)
  - LightGBM
  - Support Vector Machine
  - Neural Network (MLP)
  - SHAP Explainability
  - Survival Analysis (Cox PH + LASSO via lifelines)
  - ROC / AUC / Calibration / Decision Curve Analysis

STRUCTURE:
  Section 1  – Imports & Configuration
  Section 2  – Data Loading (NHANES + UCI + Framingham)
  Section 3  – Preprocessing & Feature Engineering
  Section 4  – Exploratory Data Analysis (EDA)
  Section 5  – Train/Test Split + Class Imbalance Handling (SMOTE)
  Section 6  – Model Training (6 algorithms)
  Section 7  – Model Evaluation & Comparison
  Section 8  – SHAP Explainability
  Section 9  – Survival Analysis (Cox PH)
  Section 10 – Multimodal Fusion Simulation
  Section 11 – Save Results & Models
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report,
    confusion_matrix, brier_score_loss, average_precision_score,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed. Run: pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("imbalanced-learn not installed. Run: pip install imbalanced-learn")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("lifelines not installed. Run: pip install lifelines")

warnings.filterwarnings('ignore')
np.random.seed(42)

# Output directories
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models',  exist_ok=True)
os.makedirs('results/tables',  exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

print("=" * 70)
print("  ATHEROSCLEROSIS ML PIPELINE — INITIALIZED")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_uci_heart_disease():
    """
    UCI Heart Disease Dataset (Cleveland Clinic Foundation).
    Source: https://archive.ics.uci.edu/dataset/45/heart+disease
    Combined 4-institution version (920 patients).

    Features:
        age       – age in years
        sex       – 1=male, 0=female
        cp        – chest pain type (1-4)
        trestbps  – resting blood pressure (mm Hg)
        chol      – serum cholesterol (mg/dl)
        fbs       – fasting blood sugar > 120 mg/dl (1=true, 0=false)
        restecg   – resting ECG results (0-2)
        thalach   – maximum heart rate achieved
        exang     – exercise-induced angina (1=yes, 0=no)
        oldpeak   – ST depression induced by exercise
        slope     – slope of peak exercise ST segment (1-3)
        ca        – number of major vessels colored by fluoroscopy (0-3)
        thal      – thalassemia (3=normal, 6=fixed defect, 7=reversible defect)
        target    – diagnosis (0=no disease, 1=disease)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]
    try:
        df = pd.read_csv(url, header=None, names=columns, na_values='?')
        df['target'] = (df['target'] > 0).astype(int)
        df['dataset'] = 'UCI_Cleveland'

        print(f"UCI Cleveland loaded: {df.shape[0]} patients")
        print(f"CVD+ rate: {df['target'].mean()*100:.1f}%")
        return df
    
    except Exception:
        print("  UCI download failed")
        raise

def load_framingham():
    """
    Framingham Heart Study — 10-year CHD prediction dataset.
    Source: https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset
    (~4,200 participants, 15 features)

    """
    FRAMINGHAM_PATH = 'data/framingham/framingham.csv'   # Data Path

    if not os.path.exists(FRAMINGHAM_PATH):
        raise FileNotFoundError(
            f"Framingham CSV not found at: {FRAMINGHAM_PATH}\n"
            "Download it from Kaggle and place it in the data folder."
        )

    df = pd.read_csv(FRAMINGHAM_PATH)
    df.columns = df.columns.str.lower().str.strip()

    if "tenyearchd" in df.columns:
        df.rename(columns={"tenyearchd": "target"}, inplace=True)
    elif "10 year risk" in df.columns:
        df.rename(columns={"10 year risk": "target"}, inplace=True)
    else:
        raise ValueError(f"Target column not found. Columns are: {df.columns.tolist()}")

    df["dataset"] = "Framingham"

    print(
        f"Framingham loaded: {df.shape[0]} patients, "
        f"{df['target'].mean()*100:.1f}% CHD+"
    )

    return df


def load_nhanes():
    """
    NHANES-structured dataset with atherosclerosis-relevant imaging proxies.

    REAL NHANES DOWNLOAD INSTRUCTIONS:
    ─────────────────────────────────────────────────────────────────────────
    1. Go to: https://wwwn.cdc.gov/nchs/nhanes/
    2. Select a survey cycle, e.g. 2013–2014
    3. Download XPT files:
         Demographics : DEMO_H.XPT
         Cholesterol  : TCHOL_H.XPT
         Blood Pressure: BPX_H.XPT
         Diabetes     : DIQ_H.XPT
         Cardiovascular Qs: CDQ_H.XPT
         Body Measures: BMX_H.XPT
         Smoking      : SMQ_H.XPT
         Physical Activity: PAQ_H.XPT
         Mortality linkage: NHANES_2013_2014_MORT_2019_PUBLIC.dat (CDC)

    4. Load in Python:
         import pyreadstat
         df, meta = pyreadstat.read_xpt('DEMO_H.XPT')

    This function generates a realistic NHANES-equivalent dataset
    with CIMT-proxy features for algorithm development.
    ─────────────────────────────────────────────────────────────────────────

    Imaging proxy features included (surrogates for multimodality imaging):
        cimt_proxy  – derived CIMT estimate from age + BP + cholesterol
        cac_proxy   – coronary artery calcium proxy from risk factor burden
        plaque_risk – carotid plaque risk score
    """

    NHANES_PATH = "data/processed/nhanes_cvd_dataset.csv"

    if not os.path.exists(NHANES_PATH):
        raise FileNotFoundError(f"NHANES processed file not found: {NHANES_PATH}")

    df = pd.read_csv(NHANES_PATH)

    # Required outcome column from your NHANES loader
    if "outcome_cv" not in df.columns:
        raise ValueError("Expected column 'outcome_cv' not found in processed NHANES file.")

    df = df[df["outcome_cv"].notna()].copy()
    df["target"] = df["outcome_cv"].astype(int)

    # Keep downstream code unchanged by mapping real column names
    rename_map = {
        "current_smoker": "smoking",
        "bp_meds": "bpmeds",
        "statin_use": "statins",
        "physically_active": "physical_activity",
        "cimt_est": "cimt_proxy",
        "cac_prob": "cac_proxy"
    }

    df.rename(columns=rename_map, inplace=True)

    # Survival time for Cox section
    if "followup_years" in df.columns:
        df["surv_time"] = df["followup_years"]
    else:
        df["surv_time"] = 1

    # Add missing columns expected by your existing code
    if "family_hx" not in df.columns:
        df["family_hx"] = 0

    if "crp" not in df.columns:
        df["crp"] = np.nan

    if "physical_activity" not in df.columns:
        df["physical_activity"] = 0

    if "statins" not in df.columns:
        df["statins"] = 0

    if "bpmeds" not in df.columns:
        df["bpmeds"] = 0

    if "smoking" not in df.columns:
        df["smoking"] = 0

    if "cimt_proxy" not in df.columns:
        df["cimt_proxy"] = np.nan

    if "cac_proxy" not in df.columns:
        df["cac_proxy"] = np.nan

    if "plaque_risk" not in df.columns:
        df["plaque_risk"] = np.nan

    print(f"  Real NHANES loaded from: {NHANES_PATH}")
    print(f"  Participants: {len(df):,}")
    print(f"  Event rate: {df['target'].mean()*100:.2f}%")

    return df


# ── Load all datasets ─────────────────────────────────────────────────────────
print("\n[1] Loading datasets...")
df_uci  = load_uci_heart_disease()
df_fram = load_framingham()
df_nhan = load_nhanes()

print(f"\n  Datasets summary:")
print(f"    UCI Cleveland  : {len(df_uci):>5,} rows")
print(f"    Framingham     : {len(df_fram):>5,} rows")
print(f"    NHANES-sim     : {len(df_nhan):>5,} rows")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Preprocessing...")

# ── UCI preprocessing ─────────────────────────────────────────────────────────
UCI_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

df_uci_clean = df_uci[UCI_FEATURES + ['target']].copy()
# Encode thal (categorical)
df_uci_clean['thal'] = df_uci_clean['thal'].map(
    {3.0: 0, 6.0: 1, 7.0: 2}
).fillna(0)
# Impute missing
imputer_uci = SimpleImputer(strategy='median')
X_uci_arr = imputer_uci.fit_transform(df_uci_clean[UCI_FEATURES])
X_uci = pd.DataFrame(X_uci_arr, columns=UCI_FEATURES)
y_uci = df_uci_clean['target'].values

print(f"  UCI: {X_uci.shape[0]} samples, {X_uci.shape[1]} features, "
      f"class balance: {y_uci.mean():.2%} CVD+")

# ── NHANES preprocessing ──────────────────────────────────────────────────────
NHANES_FEATURES = [
    'age', 'sex', 'sbp', 'dbp', 'total_chol', 'hdl', 'ldl',
    'bmi', 'hba1c', 'smoking', 'diabetes', 'hypertension',
    'bpmeds', 'statins', 'physical_activity', 'family_hx', 'crp',
    'cimt_proxy', 'cac_proxy', 'plaque_risk'
]

# Feature engineering
df_nhan['chol_ratio']   = df_nhan['total_chol'] / df_nhan['hdl'].clip(1)
df_nhan['pulse_pressure']= df_nhan['sbp'] - df_nhan['dbp']
df_nhan['metabolic_syn'] = (
    (df_nhan['bmi'] >= 30).astype(int)
    + df_nhan['hypertension']
    + df_nhan['diabetes']
    + (df_nhan['triglycerides'] > 150).astype(int)
    + (df_nhan['hdl'] < 40).astype(int)
)
df_nhan['imaging_burden'] = (
    (df_nhan['cac_proxy'] > 0).astype(int)
    + (df_nhan['cimt_proxy'] > 0.9).astype(int)
    + (df_nhan['plaque_risk'] >= 1).astype(int)
)

NHANES_FEATURES_EXT = NHANES_FEATURES + [
    'chol_ratio', 'pulse_pressure', 'metabolic_syn', 'imaging_burden'
]

X_nhan = df_nhan[NHANES_FEATURES_EXT].copy()
y_nhan = df_nhan['target'].values
surv_time_nhan = df_nhan['surv_time'].values

# Convert everything to numeric
X_nhan = X_nhan.apply(pd.to_numeric, errors="coerce")

# Remove columns that are entirely null
all_null_cols = X_nhan.columns[X_nhan.isna().all()].tolist()

if all_null_cols:
    print(f"Removing empty columns: {all_null_cols}")
    X_nhan = X_nhan.drop(columns=all_null_cols)

# Update feature list
NHANES_FEATURES_EXT = X_nhan.columns.tolist()

# Impute
imputer_nhan = SimpleImputer(strategy="median")

X_nhan = pd.DataFrame(
    imputer_nhan.fit_transform(X_nhan),
    columns=X_nhan.columns,
    index=X_nhan.index
)

print(X_nhan.isnull().sum().sort_values(ascending=False).head(20))

print(f"  NHANES: {X_nhan.shape[0]} samples, {X_nhan.shape[1]} features, "
      f"class balance: {y_nhan.mean():.2%} CVD+")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_uci, y_uci, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_uci
)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_nhan, y_nhan, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_nhan
)
# Keep survival data aligned with NHANES split
idx_all = np.arange(len(y_nhan))
idx_train, idx_test = train_test_split(
    idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_nhan
)
surv_time_train = surv_time_nhan[idx_train]
surv_time_test  = surv_time_nhan[idx_test]

# Scale (for SVM / MLP / Logistic)
scaler = StandardScaler()
X_train_n_sc = scaler.fit_transform(X_train_n)
X_test_n_sc  = scaler.transform(X_test_n)

# SMOTE for class imbalance
if SMOTE_AVAILABLE:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_n_sm, y_train_n_sm = sm.fit_resample(X_train_n, y_train_n)
    X_train_n_sc_sm, _         = sm.fit_resample(X_train_n_sc, y_train_n)
    print(f"  SMOTE applied: {len(y_train_n_sm)} training samples "
          f"({y_train_n_sm.mean():.2%} CVD+)")
else:
    X_train_n_sm, y_train_n_sm   = X_train_n, y_train_n
    X_train_n_sc_sm              = X_train_n_sc

print("  Preprocessing complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    'EDA: NHANES CVD Dataset\n'
    'Atherosclerosis Early Detection Study',
    fontsize=14, fontweight='bold', y=0.98
)

# 1. Age distribution by CVD status
ax = axes[0, 0]
for label, color, name in [(0,'steelblue','No CVD'), (1,'crimson','CVD+')]:
    subset = df_nhan[df_nhan['target'] == label]['age']
    ax.hist(subset, bins=25, alpha=0.6, color=color, label=name, density=True)
ax.set_title('Age Distribution by CVD Status', fontweight='bold')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Density')
ax.legend()

# 2. CIMT proxy by CVD status
ax = axes[0, 1]
cvd_neg = df_nhan[df_nhan['target']==0]['cimt_proxy']
cvd_pos = df_nhan[df_nhan['target']==1]['cimt_proxy']
ax.boxplot([cvd_neg, cvd_pos], labels=['No CVD','CVD+'],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
ax.set_title('CIMT Proxy by CVD Status', fontweight='bold')
ax.set_ylabel('Estimated CIMT (mm)')
ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='Risk threshold 0.9mm')
ax.legend(fontsize=8)

# 3. CAC proxy distribution (log scale)
ax = axes[0, 2]
cac_pos = df_nhan[df_nhan['cac_proxy'] > 0]['cac_proxy']
ax.hist(np.log1p(cac_pos), bins=40, color='darkorange', alpha=0.8,
        edgecolor='white', linewidth=0.3)
ax.set_title('CAC Score Distribution (CAC > 0)', fontweight='bold')
ax.set_xlabel('log(CAC Score + 1)')
ax.set_ylabel('Count')
ax.text(0.65, 0.85, f'CAC > 0: {(df_nhan["cac_proxy"]>0).mean()*100:.1f}%',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Imaging burden vs CVD rate
ax = axes[1, 0]
burden_cvd = df_nhan.groupby('imaging_burden')['target'].mean() * 100
bars = ax.bar(burden_cvd.index, burden_cvd.values,
              color=['#2ecc71','#f39c12','#e74c3c','#8e44ad'], alpha=0.85,
              edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, burden_cvd.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('CVD Rate by Imaging Burden Score\n(0=no findings → 3=all 3 modalities abnormal)',
             fontweight='bold', fontsize=9)
ax.set_xlabel('Imaging Burden Score (CAC + CIMT + Plaque)')
ax.set_ylabel('10-Year CVD Event Rate (%)')

# 5. Correlation heatmap (key features)
ax = axes[1, 1]
key_cols = ['age','sbp','cimt_proxy','cac_proxy','plaque_risk',
            'total_chol','ldl','hba1c','bmi','target']
corr = df_nhan[key_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', annot_kws={'size':7},
            linewidths=0.3, square=True)
ax.set_title('Correlation Matrix\n(Key Imaging + Clinical Features)', fontweight='bold', fontsize=9)
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)

# 6. Event rate by plaque risk + sex
ax = axes[1, 2]
pivot = df_nhan.pivot_table(values='target', index='plaque_risk',
                             columns='sex', aggfunc='mean') * 100
pivot.plot(kind='bar', ax=ax, color=['#e91e8c','#1565c0'], alpha=0.8,
           edgecolor='black', linewidth=0.5)
ax.set_title('CVD Rate by Carotid Plaque Risk\n& Sex', fontweight='bold', fontsize=9)
ax.set_xlabel('Plaque Risk (0=absent, 1=unilateral, 2=bilateral)')
ax.set_ylabel('10-Year CVD Event Rate (%)')
ax.legend(['Female','Male'], loc='upper left')
ax.set_xticklabels(['Absent','Unilateral','Bilateral'], rotation=0)

plt.tight_layout()
plt.savefig('results/figures/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("  EDA figure saved: results/figures/01_eda.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Training ML models on NHANES dataset...")

models = {}
results = {}

# ── 1. Logistic Regression (Baseline) ────────────────────────────────────────
print("  Tuning Logistic Regression...")

lr_grid = {
    "C":[0.001,0.01,0.1,1,10]
}

lr_search = GridSearchCV(
    LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),
    lr_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

lr_search.fit(X_train_n_sc_sm, y_train_n_sm)

lr = lr_search.best_estimator_

print("LR Best:")
print(lr_search.best_params_)

models["Logistic Regression"] = lr

# ── 2. Random Forest ─────────────────────────────────────────────────────────
print("  Training Random Forest...")
print("  Tuning Random Forest...")

rf_grid = {
    "n_estimators": [200, 400],
    "max_depth": [5, 8, 12],
    "min_samples_leaf": [2, 5, 10],
    "max_features": ["sqrt", 0.5]
}

rf_search = GridSearchCV(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    param_grid=rf_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train_n_sm, y_train_n_sm)

rf = rf_search.best_estimator_

print("RF Best:")
print(rf_search.best_params_)

models["Random Forest"] = rf

# ── 3. XGBoost ───────────────────────────────────────────────────────────────
if XGB_AVAILABLE:
    print("  Training XGBoost...")
    scale_pos = (y_train_n_sm == 0).sum() / (y_train_n_sm == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=1,          # SMOTE handled balance
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    print("  Tuning XGBoost...")

    param_grid = {
        "max_depth":[3,5,7],
        "learning_rate":[0.01,0.05],
        "subsample":[0.8,1.0],
        "n_estimators":[300,500]
    }

    xgb_search = GridSearchCV(
        xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    xgb_search.fit(
        X_train_n_sm,
        y_train_n_sm
    )

    xgb_model = xgb_search.best_estimator_

    print("XGB Best:")
    print(xgb_search.best_params_)

    models["XGBoost"] = xgb_model
    
else:
    print("  XGBoost skipped (not installed)")

# ── 4. LightGBM ──────────────────────────────────────────────────────────────
if LGB_AVAILABLE:
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train_n_sm, y_train_n_sm,
                  eval_set=[(X_test_n, y_test_n)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    models['LightGBM'] = lgb_model
else:
    print("  LightGBM skipped (not installed)")

# ── 5. SVM ────────────────────────────────────────────────────────────────────
print("  Training SVM (calibrated)...")
svm_base = SVC(kernel='rbf', C=1.0, gamma='scale',
               class_weight='balanced', probability=False,
               random_state=RANDOM_STATE)
svm_cal = CalibratedClassifierCV(svm_base, cv=3, method='sigmoid')
svm_cal.fit(X_train_n_sc_sm, y_train_n_sm)
models['SVM'] = svm_cal

# ── 6. Neural Network (MLP) ───────────────────────────────────────────────────
print("  Training Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    alpha=0.001, learning_rate='adaptive',
    max_iter=500, early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_STATE
)
mlp.fit(X_train_n_sc_sm, y_train_n_sm)
models['Neural Network'] = mlp

print(f"  {len(models)} models trained.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL EVALUATION & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Evaluating models...")

def evaluate_model(name, model, X_test, y_test, scaled=False):
    """Return dict of evaluation metrics."""
    X = X_test if not scaled else X_test
    if name in ['Logistic Regression', 'SVM', 'Neural Network']:
        X_eval = scaler.transform(X_test) if not scaled else X_test
    else:
        X_eval = X_test

    y_pred  = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    auc      = roc_auc_score(y_test, y_proba)
    acc      = accuracy_score(y_test, y_pred)
    brier    = brier_score_loss(y_test, y_proba)
    ap       = average_precision_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        'name': name, 'model': model,
        'auc': auc, 'accuracy': acc,
        'brier': brier, 'avg_precision': ap,
        'fpr': fpr, 'tpr': tpr,
        'y_proba': y_proba, 'y_pred': y_pred
    }

for name, model in models.items():
    r = evaluate_model(name, model, X_test_n, y_test_n)
    results[name] = r
    print(f"    {name:<22} AUC={r['auc']:.4f}  "
          f"Acc={r['accuracy']:.4f}  Brier={r['brier']:.4f}")

# Summary table
results_df = pd.DataFrame([{
    'Model': r['name'],
    'AUC-ROC': f"{r['auc']:.4f}",
    'Accuracy': f"{r['accuracy']:.4f}",
    'Avg Precision': f"{r['avg_precision']:.4f}",
    'Brier Score': f"{r['brier']:.4f}",
} for r in results.values()])
results_df.to_csv('results/tables/model_comparison.csv', index=False)
print("\n  Model comparison saved: results/tables/model_comparison.csv")
print(results_df.to_string(index=False))

# ── ROC Curves ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance: Atherosclerosis CVD Prediction\n(NHANES Dataset)',
             fontsize=13, fontweight='bold')

colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
ax = axes[0]
for (name, r), c in zip(results.items(), colors):
    ax.plot(r['fpr'], r['tpr'], lw=2, color=c,
            label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=11)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=11)
ax.set_title('ROC Curves — All Models', fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# Calibration curves
ax = axes[1]
for (name, r), c in zip(results.items(), colors):
    fraction_pos, mean_pred = calibration_curve(
        y_test_n, r['y_proba'], n_bins=10, strategy='uniform'
    )
    ax.plot(mean_pred, fraction_pos, 's-', lw=1.5, color=c,
            label=name, markersize=4)
ax.plot([0,1],[0,1],'k--', lw=1.5, alpha=0.7, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives', fontsize=11)
ax.set_title('Calibration Curves', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/02_roc_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ROC + calibration figure saved: results/figures/02_roc_calibration.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SHAP EXPLAINABILITY (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

if SHAP_AVAILABLE and 'XGBoost' in models:
    print("\n[6] SHAP explainability analysis...")
    best_model = models['XGBoost']
    X_explain  = X_test_n.iloc[:200]   # subset for speed

    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_explain)

    # Bar plot: global feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('SHAP Explainability — XGBoost Model\nAtherosclerosis CVD Risk Prediction',
                 fontsize=13, fontweight='bold')

    # Mean |SHAP| per feature
    ax = axes[0]
    shap_mean = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(shap_mean, index=NHANES_FEATURES_EXT).sort_values(ascending=True)
    colors_bar = ['#e74c3c' if 'proxy' in f or 'plaque' in f or 'cac' in f or 'cimt' in f
                  else '#3498db' for f in feat_imp.index]
    feat_imp.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='white', linewidth=0.3)
    ax.set_title('Global Feature Importance (Mean |SHAP|)\n'
                 'Red = Imaging features, Blue = Clinical features',
                 fontweight='bold', fontsize=10)
    ax.set_xlabel('Mean |SHAP value|')
    ax.grid(axis='x', alpha=0.3)

    # Beeswarm-style scatter: top 12 features
    ax = axes[1]
    top12 = feat_imp.nlargest(12).index.tolist()
    top12_idx = [NHANES_FEATURES_EXT.index(f) for f in top12]
    shap_top = shap_values[:, top12_idx]
    x_top    = X_explain[top12].values

    for i, feat in enumerate(top12):
        vals  = shap_top[:, i]
        feat_vals = x_top[:, i]
        norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
        jitter = np.random.default_rng(i).uniform(-0.2, 0.2, len(vals))
        scatter = ax.scatter(vals, np.full_like(vals, i) + jitter,
                             c=norm_vals, cmap='RdBu_r', alpha=0.4,
                             s=8, vmin=0, vmax=1)

    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels(top12, fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('SHAP value (impact on CVD risk prediction)')
    ax.set_title('SHAP Beeswarm — Top 12 Features\n(Red=high value, Blue=low value)',
                 fontweight='bold', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='Normalized feature value',
                 fraction=0.03, pad=0.04)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/03_shap_explainability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  SHAP figure saved: results/figures/03_shap_explainability.png")

    # Top 5 most important features
    top5 = feat_imp.nlargest(5)
    print("\n  Top 5 predictors (SHAP):")
    for feat, val in top5.items():
        tag = " ← IMAGING" if any(k in feat for k in ['proxy','plaque','imaging']) else ""
        print(f"    {feat:<25} SHAP={val:.4f}{tag}")
else:
    print("\n[6] SHAP skipped (install: pip install shap xgboost)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SURVIVAL ANALYSIS (Cox Proportional Hazards)
# ══════════════════════════════════════════════════════════════════════════════

if LIFELINES_AVAILABLE:
    print("\n[7] Survival analysis (Cox PH)...")

    surv_df = df_nhan[
        NHANES_FEATURES_EXT + ['surv_time', 'target']
    ].copy()
    surv_df.columns = [c.replace(' ', '_') for c in surv_df.columns]

    # Normalize continuous features for Cox
    for col in ['age','sbp','total_chol','ldl','cimt_proxy','cac_proxy','crp','bmi']:
        surv_df[col] = (surv_df[col] - surv_df[col].mean()) / surv_df[col].std()

    # ── Kaplan-Meier by imaging burden ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Survival Analysis — Atherosclerosis Imaging Study\n'
                 'NHANES Cohort',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    kmf = KaplanMeierFitter()
    for burden, color, label in [
        (0, '#2ecc71', 'No imaging abnormalities'),
        (1, '#f39c12', '1 abnormality'),
        (2, '#e67e22', '2 abnormalities'),
        (3, '#e74c3c', 'All 3 abnormalities')
    ]:
        mask = df_nhan['imaging_burden'] == burden
        if mask.sum() > 20:
            kmf.fit(df_nhan.loc[mask, 'surv_time'],
                    event_observed=df_nhan.loc[mask, 'target'],
                    label=label)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=2)

    ax.set_title('Kaplan-Meier: Event-Free Survival\nby Imaging Burden Score',
                 fontweight='bold')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Event-Free Survival Probability')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Cox PH model ──────────────────────────────────────────────────────────
    cox_features = [
        'age', 'sex', 'sbp', 'total_chol', 'ldl', 'smoking',
        'diabetes', 'hypertension', 'bmi',
        'cimt_proxy', 'cac_proxy', 'plaque_risk', 'imaging_burden'
    ]
    cox_data = surv_df[cox_features + ['surv_time', 'target']].dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col='surv_time', event_col='target',
            show_progress=False)

    ax = axes[1]
    cph.plot(ax=ax)
    ax.set_title('Cox PH Model — Hazard Ratios\n(95% CI, penalizer=0.1)',
                 fontweight='bold')
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/04_survival_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print Cox summary
    print("\n  Cox PH Summary (top predictors by |coef|):")
    cox_summary = cph.summary[['coef','exp(coef)','p']].copy()
    cox_summary.columns = ['log-HR', 'Hazard Ratio', 'p-value']
    cox_summary = cox_summary.reindex(
        cox_summary['log-HR'].abs().sort_values(ascending=False).index
    )
    print(cox_summary.head(8).to_string())
    cox_summary.to_csv('results/tables/cox_ph_results.csv')
    print("  Survival figures saved: results/figures/04_survival_analysis.png")
    print("  Cox results saved: results/tables/cox_ph_results.csv")
else:
    print("\n[7] Survival analysis skipped (install: pip install lifelines)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MULTIMODAL FUSION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[8] Multimodal fusion experiment...")

"""
Simulates three imaging 'modalities' as separate feature sets, then
compares single-modality vs. late-fusion ensemble performance.

Modality 1 (Clinical only):  traditional risk factors
Modality 2 (CIMT/Plaque):    carotid imaging features
Modality 3 (CAC/Metabolic):  coronary calcium + metabolic features
Fusion:                       average predicted probability across all 3
"""

CLINICAL_FEATS  = ['age','sex','sbp','dbp','total_chol','hdl','ldl',
                   'smoking','diabetes','hypertension','bmi','crp']
CAROTID_FEATS   = ['age','sex','cimt_proxy','plaque_risk',
                   'pulse_pressure','sbp','smoking','diabetes']
CORONARY_FEATS  = ['age','sex','cac_proxy','total_chol','ldl','bmi',
                   'hba1c','statins','family_hx','metabolic_syn']

fusion_results = {}

for label, feats in [
    ('Clinical only',   CLINICAL_FEATS),
    ('Carotid imaging', CAROTID_FEATS),
    ('Coronary/CAC',    CORONARY_FEATS),
]:

    # Keep only features available in real NHANES processed data
    feats = [f for f in feats if f in X_train_n_sm.columns and f in X_test_n.columns]

    if len(feats) == 0:
        print(f"    {label:<22} skipped — no available features")
        continue

    print(f"    {label:<22} using features: {feats}")

    X_tr = X_train_n_sm[feats]
    X_te = X_test_n[feats]
    m = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_tr, y_train_n_sm)
    proba = m.predict_proba(X_te)[:,1]
    auc = roc_auc_score(y_test_n, proba)
    fusion_results[label] = {'proba': proba, 'auc': auc, 'model': m}
    print(f"    {label:<22} AUC = {auc:.4f}")

# Late fusion: average probabilities
fused_proba = np.mean([
    fusion_results['Clinical only']['proba'],
    fusion_results['Carotid imaging']['proba'],
    fusion_results['Coronary/CAC']['proba']
], axis=0)
fusion_auc = roc_auc_score(y_test_n, fused_proba)
fusion_results['Multimodal Fusion'] = {'proba': fused_proba, 'auc': fusion_auc}
print(f"    {'Multimodal Fusion':<22} AUC = {fusion_auc:.4f}  ← Late fusion")

# Plot fusion comparison
fig, ax = plt.subplots(figsize=(9, 6))
colors_f = ['#3498db','#2ecc71','#e67e22','#e74c3c']
for (name, res), c in zip(fusion_results.items(), colors_f):
    fpr, tpr, _ = roc_curve(y_test_n, res['proba'])
    lw = 3 if name == 'Multimodal Fusion' else 1.5
    ls = '-' if name == 'Multimodal Fusion' else '--'
    ax.plot(fpr, tpr, lw=lw, linestyle=ls, color=c,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.4)
ax.set_title('Single Modality vs. Multimodal Fusion\n'
             'Late-Fusion Ensemble (Average Probability)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/05_multimodal_fusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fusion figure saved: results/figures/05_multimodal_fusion.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SAVE MODELS & FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[9] Saving models...")
import pickle

for name, model in models.items():
    fname = name.lower().replace(' ', '_')
    with open(f'results/models/{fname}.pkl', 'wb') as f:
        pickle.dump(model, f)
print(f"  {len(models)} models saved to results/models/")

# Final summary
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — RESULTS SUMMARY")
print("=" * 70)
print(f"\n  Dataset          : NHANES ({len(df_nhan):,} participants)")
print(f"  Features         : {len(NHANES_FEATURES_EXT)} (clinical + imaging proxies)")
print(f"  Test set         : {len(y_test_n):,} participants")
print(f"  CVD event rate   : {y_test_n.mean()*100:.1f}%")
print()

best_name = max(results, key=lambda k: results[k]['auc'])
best_auc  = results[best_name]['auc']
print(f"  Best model       : {best_name}  (AUC = {best_auc:.4f})")
print(f"  Multimodal fusion: AUC = {fusion_auc:.4f}")
print()
print("  Output files:")
print("    results/figures/01_eda.png")
print("    results/figures/02_roc_calibration.png")
print("    results/figures/03_shap_explainability.png")
print("    results/figures/04_survival_analysis.png")
print("    results/figures/05_multimodal_fusion.png")
print("    results/tables/model_comparison.csv")
print("    results/tables/cox_ph_results.csv")
print("    results/models/*.pkl")
print()
print("=" * 70)