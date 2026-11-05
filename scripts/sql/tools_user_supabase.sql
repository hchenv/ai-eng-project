CREATE USER tools_user WITH PASSWORD 'ewTFDFew';


-- Only run this once the schemas and the tables are created
grant all on schema shopping_carts to tools_user;
grant all on all tables in schema shopping_carts to tools_user;
grant all on all sequences in schema shopping_carts to tools_user;

-- Only run this once the schemas and the tables are created
grant all on schema warehouses to tools_user;
grant all on all tables in schema warehouses to tools_user;
grant all on all sequences in schema warehouses to tools_user;