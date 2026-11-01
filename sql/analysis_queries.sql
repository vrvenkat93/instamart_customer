-- Seven analytical queries (run against the normalized analytical tables created by Part A).
-- 1. Monthly revenue and order count
SELECT substr(order_date, 1, 7) AS month, COUNT(DISTINCT order_id) AS orders,
       ROUND(SUM(amount), 2) AS revenue
FROM _orders GROUP BY month ORDER BY month;

-- 2. Top products (falls back to the "All orders" product if no line-item source exists)
SELECT product_name, SUM(quantity_value) AS units, ROUND(SUM(amount_value), 2) AS revenue
FROM _items GROUP BY product_name ORDER BY revenue DESC LIMIT 10;

-- 3. Highest lifetime-value customers
SELECT customer_id, COUNT(DISTINCT order_id) AS order_count, ROUND(SUM(amount), 2) AS lifetime_value
FROM _orders GROUP BY customer_id ORDER BY lifetime_value DESC LIMIT 20;

-- 4. Revenue by inferred city/state location
SELECT location, COUNT(DISTINCT customer_id) AS customers, ROUND(SUM(amount), 2) AS revenue
FROM _orders GROUP BY location ORDER BY revenue DESC;

-- 5. Payment method mix
SELECT payment_method, COUNT(*) AS orders, ROUND(SUM(amount), 2) AS revenue
FROM _orders GROUP BY payment_method ORDER BY orders DESC;

-- 6. Order status distribution
SELECT status, COUNT(*) AS orders FROM _orders GROUP BY status ORDER BY orders DESC;

-- 7. Registered customers without an order
SELECT c.customer_id FROM _customers c
LEFT JOIN _orders o ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL ORDER BY c.customer_id;
