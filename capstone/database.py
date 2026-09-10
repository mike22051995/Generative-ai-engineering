import psycopg2
from psycopg2 import pool
import psycopg2.extras
from config import settings

connection_pool=psycopg2.pool.SimpleConnectionPool(
    minconn=2,
    maxconn=10,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    dbname=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD
)

def get_connection():
    """Borrow a connection from pool"""
    return connection_pool.getconn()

def release_connection(conn):
    """return connection back to pool."""
    connection_pool.putconn(conn)

def execute_query(query:str,params:tuple=None)->list:
    """
    Execute a SELECT query and return results.
    Automatically  borrows and returns connection.
    """
    conn=get_connection()
    try:
        cursor=conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query,params)
        results=cursor.fetchall()
        cursor.close()
        return [dict(row) for row in results]
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)

def execute_write(query:str,params:tuple=None)->None:
    """
    Execute INSERT/UPDATE/DELETE query.
    Automatically handles commit and rollback.
    """
    conn=get_connection()
    try:
        cursor=conn.cursor()
        cursor.execute(query,params)
        conn.commit()
        cursor.close()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)
