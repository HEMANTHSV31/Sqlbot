USE chatbot_db;

-- Insert sample users
INSERT INTO users (name, email) VALUES
('John Doe', 'john.doe@email.com'),
('Jane Smith', 'jane.smith@email.com'),
('Bob Johnson', 'bob.johnson@email.com'),
('Alice Brown', 'alice.brown@email.com'),
('Charlie Wilson', 'charlie.wilson@email.com');

-- Insert sample products
INSERT INTO products (name, sku, price) VALUES
('Laptop Pro', 'LP-1001', 1299.99),
('Wireless Mouse', 'WM-2001', 29.99),
('Mechanical Keyboard', 'MK-3001', 89.99),
('Monitor 24"', 'MN-4001', 199.99),
('Webcam HD', 'WC-5001', 49.99);

-- Insert sample orders
INSERT INTO orders (user_id, total) VALUES
(1, 1329.98),
(2, 119.98),
(3, 289.98),
(4, 49.99),
(1, 89.99);

-- Insert sample order items
INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
(2, 2, 2, 29.99),
(2, 4, 1, 199.99),
(3, 3, 2, 89.99),
(3, 5, 1, 49.99),
(4, 5, 1, 49.99),
(5, 3, 1, 89.99);

-- Insert sample product sales
INSERT INTO product_sales (product_id, sale_date, units_sold, revenue) VALUES
(1, '2024-01-15', 5, 6499.95),
(2, '2024-01-15', 20, 599.80),
(3, '2024-01-15', 8, 719.92),
(4, '2024-01-15', 3, 599.97),
(5, '2024-01-15', 12, 599.88),
(1, '2024-01-16', 3, 3899.97),
(2, '2024-01-16', 15, 449.85);