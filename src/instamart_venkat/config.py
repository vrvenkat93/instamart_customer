from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
POLICY_DIR = ROOT / "data" / "policies"
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"
DB_PATH = ARTIFACTS / "instamart.db"
RANDOM_STATE = 42

for directory in (DATA_RAW, DATA_PROCESSED, POLICY_DIR, ARTIFACTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

# Canonical field -> common source header spellings.
ALIASES = {
    "customer_id": ["customer_id", "customerid", "cust_id", "user_id", "userid"],
    "order_id": ["order_id", "orderid", "id", "transaction_id"],
    "product_id": ["product_id", "productid", "sku", "item_id"],
    "ticket_id": ["ticket_id", "ticketid", "case_id", "support_id"],
    "order_date": ["order_date", "orderdate", "created_at", "created_date", "purchase_date", "date"],
    "order_timestamp": ["order_timestamp", "timestamp", "ordered_at", "created_at"],
    "amount": ["amount", "total", "total_amount", "order_amount", "net_amount", "revenue", "price"],
    "quantity": ["quantity", "qty", "units"],
    "status": ["status", "order_status", "ticket_status"],
    "email": ["email", "email_address", "customer_email"],
    "phone": ["phone", "phone_number", "mobile", "mobile_number"],
    "text": ["text", "ticket_text", "message", "description", "subject", "complaint"],
    "category": ["category", "ticket_category", "issue_type"],
    "payment_method": ["payment_method", "payment_type", "payment_mode"],
    "city": ["city", "town"],
    "state": ["state", "province"],
    "product_name": ["product_name", "name", "item_name"],
    "unit_price": ["unit_price", "selling_price", "price"],
    "refund_amount": ["refund_amount", "requested_refund", "amount_requested"],
}
