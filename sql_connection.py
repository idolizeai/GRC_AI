import pyodbc
from datetime import datetime

def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=216.48.191.98;'
        'DATABASE=GRC_AI;'
        'UID=ibsadmin;'
        'PWD=Viking@@ibs2023'
    )
    return conn
