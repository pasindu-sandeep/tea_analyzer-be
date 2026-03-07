import psycopg2

DB_HOST = "34.136.45.26"
DB_NAME = "tea-analyzer"
DB_USER = "postgres"
DB_PASS = "Nimna654321@"
DB_PORT = 5432

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
