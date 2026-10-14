"""
Lightweight baseline evaluator for DINE MATE (minimal dependencies)
- Health ML: trains a RandomForest on `info/health_tracking_20k.csv` and reports accuracy/precision/recall/F1/AUC (if sklearn available)
- Recommender (popularity baseline): builds a train/test split from `info/amazon_reviews_20k.csv` and reports hit@10 using top-popular items from train

This script avoids heavy packages (no surprise, no torch). If pandas/scikit-learn are missing it prints a short message.

Run:
    python eval_baseline_light.py

Outputs a small JSON report to `cache/baseline_report_light.json`
"""
import json
from pathlib import Path
import sys

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except Exception as e:
    print("Missing required Python packages. Please install them and re-run this script:")
    print("    pip install pandas numpy scikit-learn")
    print(f"Detailed error: {e}")
    sys.exit(1)

report = {'health': {}, 'recommender_popularity': {}}
root = Path(__file__).parent
info = root / 'info'

# --- Health baseline ---
health_path = info / 'health_tracking_20k.csv'
if health_path.exists():
    try:
        df = pd.read_csv(health_path)
        # Try to find a binary target column (values only 0/1)
        binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]
        if binary_cols:
            target_col = binary_cols[0]
        else:
            # Try name-based detection
            target_cols = [c for c in df.columns if 'diabetes' in c.lower() or c.lower() in ('label','target','has_condition')]
            if target_cols:
                target_col = target_cols[0]
            else:
                # fallback: attempt binary target using 'bmi' threshold (example)
                if 'bmi' in df.columns:
                    df['__target__'] = (df['bmi'] > 25).astype(int)
                    target_col = '__target__'
                else:
                    raise ValueError('No obvious binary target found in health dataset')

        # Prepare features - drop identifiers and target
        X = df.select_dtypes(include=[np.number]).copy()
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        y = df[target_col].copy()

        # Simple imputation for numeric cols
        X = X.fillna(X.median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            y_prob = clf.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        report['health'] = {
            'target_column': target_col,
            'n_samples': int(len(df)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None
        }
        print('Health baseline metrics:')
        print(json.dumps(report['health'], indent=2))
    except Exception as e:
        print('Health baseline failed:', e)
        report['health'] = {'error': str(e)}
else:
    print('Health dataset not found at info/health_tracking_20k.csv')
    report['health'] = {'error': 'dataset missing'}

# --- Popularity recommender baseline (hit@10) ---
reviews_path = info / 'amazon_reviews_20k.csv'
if reviews_path.exists():
    try:
        reviews = pd.read_csv(reviews_path)
        # Try common column names
        user_col = None
        item_col = None
        for c in reviews.columns:
            if c.lower() in ('userid','user_id','userId'.lower()):
                user_col = c
            if c.lower() in ('productname','product_name','productid','productId','product'):
                item_col = c
        # fallback heuristics
        if user_col is None:
            for c in reviews.columns:
                if 'user' in c.lower():
                    user_col = c
                    break
        if item_col is None:
            for c in reviews.columns:
                if 'product' in c.lower() or 'item' in c.lower() or 'title' in c.lower() or 'name' in c.lower():
                    item_col = c
                    break
        if user_col is None or item_col is None:
            raise ValueError('Could not detect user/item columns in reviews dataset')

        # Build simple per-user holdout: for each user with >=2 interactions, hold out one item
        reviews = reviews[[user_col, item_col]].dropna()
        grp = reviews.groupby(user_col)
        train_rows = []
        test_rows = []
        for u, g in grp:
            if len(g) < 2:
                # single interaction -> keep in train
                train_rows.extend(g.index.tolist())
                continue
            gidx = g.index.tolist()
            test_idx = gidx[-1]
            train_idx = gidx[:-1]
            test_rows.append(test_idx)
            train_rows.extend(train_idx)
        train_df = reviews.loc[train_rows]
        test_df = reviews.loc[test_rows]

        # Popular items from train
        popular = train_df[item_col].value_counts().head(100).index.tolist()

        # Evaluate hit@10: for each test user, see if held-out item is in top-10 popular
        hits = 0
        total = 0
        for idx, row in test_df.iterrows():
            total += 1
            if row[item_col] in popular[:10]:
                hits += 1
        hit10 = hits / total if total > 0 else None
        report['recommender_popularity'] = {
            'n_users_tested': int(total),
            'hit@10': float(hit10) if hit10 is not None else None
        }
        print('Recommender popularity baseline:')
        print(json.dumps(report['recommender_popularity'], indent=2))
    except Exception as e:
        print('Recommender baseline failed:', e)
        report['recommender_popularity'] = {'error': str(e)}
else:
    print('Reviews dataset not found at info/amazon_reviews_20k.csv')
    report['recommender_popularity'] = {'error': 'dataset missing'}

# Save report
report_path = CACHE_DIR / 'baseline_report_light.json'
report_path.write_text(json.dumps(report, indent=2))
print('\nSaved baseline report to', report_path)
