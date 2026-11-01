import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import ALIASES, DATA_PROCESSED, DB_PATH, ARTIFACTS
from .utils import clean_frame, load_csvs, resolve_column, safe_table_name, write_json

SQL_ANALYSES = {
    "01_monthly_revenue": """
        SELECT substr(order_date,1,7) AS month, COUNT(DISTINCT order_id) AS orders,
               ROUND(SUM(amount),2) AS revenue
        FROM _orders GROUP BY month ORDER BY month
    """,
    "02_top_products": """
        SELECT product_name, SUM(quantity_value) AS units,
               ROUND(SUM(amount_value),2) AS revenue
        FROM _items GROUP BY product_name ORDER BY revenue DESC LIMIT 10
    """,
    "03_customer_value": """
        SELECT customer_id, COUNT(DISTINCT order_id) AS order_count,
               ROUND(SUM(amount),2) AS lifetime_value
        FROM _orders GROUP BY customer_id ORDER BY lifetime_value DESC LIMIT 20
    """,
    "04_location_summary": """
        SELECT location, COUNT(DISTINCT customer_id) AS customers,
               ROUND(SUM(amount),2) AS revenue
        FROM _orders GROUP BY location ORDER BY revenue DESC
    """,
    "05_payment_mix": """
        SELECT payment_method, COUNT(*) AS orders, ROUND(SUM(amount),2) AS revenue
        FROM _orders GROUP BY payment_method ORDER BY orders DESC
    """,
    "06_order_status": """
        SELECT status, COUNT(*) AS orders FROM _orders GROUP BY status ORDER BY orders DESC
    """,
    "07_customers_without_orders": """
        SELECT c.customer_id FROM _customers c
        LEFT JOIN _orders o ON c.customer_id=o.customer_id
        WHERE o.customer_id IS NULL ORDER BY c.customer_id
    """,
}

def _pick_table(tables, tokens):
    matches = [(name, df) for name, df in tables.items() if any(token in name for token in tokens)]
    if not matches:
        return None, None
    # Prefer a primary entity table over line-item/detail tables.
    matches.sort(key=lambda pair: (any(x in pair[0] for x in ("item", "line", "detail")), len(pair[0])))
    return matches[0]

def _canonical_orders(tables):
    name, df = _pick_table(tables, ("order", "transaction", "purchase"))
    if df is None:
        raise ValueError("Could not identify an orders CSV (filename should include 'order' or 'transaction').")
    mapping = {k: resolve_column(df, k, ALIASES) for k in ("order_id","customer_id","order_date","amount","status","payment_method","city","state")}
    required = ("order_id","customer_id","order_date","amount")
    missing = [k for k in required if not mapping[k]]
    if missing:
        raise ValueError(f"Orders table {name!r} is missing required fields: {missing}. Available: {list(df.columns)}")
    out = pd.DataFrame()
    for key, col in mapping.items():
        if col:
            out[key] = df[col]
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0)
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce", utc=True)
    out = out.dropna(subset=["order_id","customer_id","order_date"])
    if "status" in out:
        # Completed status filtering is applied downstream; retain source status here.
        out["status"] = out["status"].astype(str).str.strip().str.lower()
    else:
        out["status"] = "completed"
    if "payment_method" not in out: out["payment_method"] = "unknown"
    if "city" in out and "state" in out:
        out["location"] = out["city"].fillna("").astype(str) + ", " + out["state"].fillna("").astype(str)
    elif "city" in out: out["location"] = out["city"].fillna("Unknown")
    elif "state" in out: out["location"] = out["state"].fillna("Unknown")
    else: out["location"] = "Unknown"
    return name, out

def run_part_a(data_dir: Path):
    raw_tables = load_csvs(data_dir)
    cleaned, quality = {}, {}
    for name, df in raw_tables.items():
        cleaned_df, report = clean_frame(df)
        cleaned[name] = cleaned_df
        quality[name] = report
        cleaned_df.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    with sqlite3.connect(DB_PATH) as conn:
        for name, df in cleaned.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        # Build stable analytical views/tables from inferred source schema.
        order_source, orders = _canonical_orders(cleaned)
        orders.to_sql("_orders", conn, if_exists="replace", index=False)
        customer_name, customers = _pick_table(cleaned, ("customer", "user", "profile"))
        if customers is not None:
            cid = resolve_column(customers, "customer_id", ALIASES)
            if cid:
                customers[[cid]].rename(columns={cid:"customer_id"}).drop_duplicates().to_sql("_customers", conn, if_exists="replace", index=False)
        else:
            pd.DataFrame({"customer_id": orders["customer_id"].drop_duplicates()}).to_sql("_customers", conn, if_exists="replace", index=False)

        item_name, items = _pick_table(cleaned, ("item", "line", "product"))
        if items is not None:
            pid = resolve_column(items, "product_name", ALIASES)
            qty = resolve_column(items, "quantity", ALIASES)
            amt = resolve_column(items, "amount", ALIASES) or resolve_column(items, "unit_price", ALIASES)
            oid = resolve_column(items, "order_id", ALIASES)
            if pid and qty and amt:
                norm_items = pd.DataFrame({"product_name":items[pid].astype(str),
                                           "quantity_value":pd.to_numeric(items[qty], errors="coerce").fillna(0),
                                           "amount_value":pd.to_numeric(items[amt], errors="coerce").fillna(0)})
                norm_items.to_sql("_items", conn, if_exists="replace", index=False)
        # A fallback item table based on orders keeps query execution explicit.
        if not items is not None or not ("_items" in [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]):
            orders.assign(product_name="All orders", quantity_value=1, amount_value=orders["amount"])[["product_name","quantity_value","amount_value"]].to_sql("_items", conn, if_exists="replace", index=False)

        outputs = {}
        for query_name, sql in SQL_ANALYSES.items():
            try:
                outputs[query_name] = pd.read_sql_query(sql, conn).to_dict(orient="records")
            except Exception as exc:
                outputs[query_name] = {"status":"SKIPPED", "reason":str(exc)}
    write_json(ARTIFACTS / "data_quality_report.json", quality)
    write_json(ARTIFACTS / "sql_query_results.json", outputs)
    print(f"[Part A] Cleaned {len(cleaned)} CSV(s); SQLite: {DB_PATH}")
    for name, result in outputs.items():
        status = "OK" if isinstance(result, list) else result["status"]
        print(f"  {name}: {status}")
    return cleaned
