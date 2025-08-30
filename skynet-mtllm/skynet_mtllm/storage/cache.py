import sqlite3

class SqliteCache:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key):
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO cache(key,value) VALUES(?,?)", (key, value))
        self.conn.commit()
