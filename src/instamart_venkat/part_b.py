import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DB_PATH, ARTIFACTS, RANDOM_STATE
from .utils import write_json

def run_part_b():
    with sqlite3.connect(DB_PATH) as conn:
        orders = pd.read_sql_query("SELECT * FROM _orders", conn)
    if orders.empty:
        raise ValueError("No valid orders available for RFM.")
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce", utc=True)
    orders["amount"] = pd.to_numeric(orders["amount"], errors="coerce").fillna(0)
    if "status" in orders:
        excluded = {"cancelled","canceled","failed","returned","refunded"}
        completed = orders[~orders["status"].astype(str).str.lower().isin(excluded)].copy()
    else:
        completed = orders.copy()
    if completed.empty: completed = orders.copy()
    snapshot = completed["order_date"].max() + pd.Timedelta(days=1)
    rfm = completed.groupby("customer_id").agg(
        last_order=("order_date","max"),
        frequency=("order_id","nunique"),
        monetary=("amount","sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order"]).dt.days.clip(lower=0)
    rfm = rfm[["customer_id","recency","frequency","monetary"]].replace([np.inf,-np.inf], np.nan).dropna()
    rfm[["frequency","monetary"]] = rfm[["frequency","monetary"]].clip(lower=0)
    if len(rfm) < 2:
        rfm["cluster"] = 0
        rfm["pca_1"] = 0.0
        rfm["pca_2"] = 0.0
        summary = rfm.groupby("cluster").agg(customers=("customer_id","count"), mean_recency=("recency","mean"),
                                             mean_frequency=("frequency","mean"), mean_monetary=("monetary","mean")).reset_index()
        write_json(ARTIFACTS/"lapse_model_metrics.json", {"status":"SKIPPED","reason":"Need at least 2 customers for supervised modelling."})
    else:
        features = rfm[["recency","frequency","monetary"]].astype(float)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)
        n_clusters = min(4, max(2, int(np.sqrt(len(rfm)/2))), len(rfm))
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        rfm["cluster"] = km.fit_predict(x_scaled)
        pca = PCA(n_components=2 if min(x_scaled.shape) >= 2 else 1, random_state=RANDOM_STATE)
        pcs = pca.fit_transform(x_scaled)
        rfm["pca_1"] = pcs[:,0]
        rfm["pca_2"] = pcs[:,1] if pcs.shape[1] > 1 else 0.0
        summary = rfm.groupby("cluster").agg(customers=("customer_id","count"), mean_recency=("recency","mean"),
                                             mean_frequency=("frequency","mean"), mean_monetary=("monetary","mean")).reset_index()
        # Transparent proxy label: high recency relative to the cohort's 75th percentile.
        # This is not a future-observed churn label.
        cutoff = float(rfm["recency"].quantile(.75))
        y = (rfm["recency"] >= cutoff).astype(int)
        X = features.copy()
        metrics = {"label_definition":f"recency >= cohort 75th percentile ({cutoff:.2f} days)",
                   "warning":"Observed-lapse proxy, not future churn ground truth."}
        if y.nunique() < 2 or y.value_counts().min() < 2:
            metrics["status"] = "SKIPPED"
            metrics["reason"] = "Insufficient examples in both classes for stratified train/test."
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE, stratify=y)
            model_specs = {
                "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                "random_forest": RandomForestClassifier(n_estimators=250, max_depth=6, class_weight="balanced",
                                                        random_state=RANDOM_STATE, n_jobs=-1)
            }
            for name, model in model_specs.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:,1]
                metrics[name] = {
                    "accuracy":float(accuracy_score(y_test,pred)),
                    "precision":float(precision_score(y_test,pred,zero_division=0)),
                    "recall":float(recall_score(y_test,pred,zero_division=0)),
                    "f1":float(f1_score(y_test,pred,zero_division=0)),
                    "roc_auc":float(roc_auc_score(y_test,proba)) if y_test.nunique()==2 else None
                }
                joblib.dump(model, ARTIFACTS/f"{name}.joblib")
            metrics["status"] = "OK"
        write_json(ARTIFACTS/"lapse_model_metrics.json", metrics)
    rfm.to_csv(ARTIFACTS/"rfm_customers.csv", index=False)
    summary.to_csv(ARTIFACTS/"rfm_cluster_summary.csv", index=False)
    rfm[["customer_id","pca_1","pca_2","cluster"]].to_csv(ARTIFACTS/"rfm_pca.csv", index=False)
    print(f"[Part B] RFM rows={len(rfm)}; clusters={rfm['cluster'].nunique()}")
    return rfm
