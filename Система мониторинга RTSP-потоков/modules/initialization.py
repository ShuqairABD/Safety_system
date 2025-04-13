import os
import sqlite3
import nest_asyncio

class Initialization:
    def __init__(self):
        # Разрешать запуск `asyncio` в (уже) работающем цикле событий
        nest_asyncio.apply()

        # Создание папки для нарушений
        os.makedirs("violation_frame", exist_ok=True)
        print("✅ Папка 'violation_frame' создана или уже существует.")

        # Создание БД SQLite
        self.DB_PATH = "violations.db"
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                violation_type TEXT,
                timestamp DATETIME
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Таблица violations создана или уже существует в базе данных.")