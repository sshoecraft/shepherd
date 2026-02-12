#!/usr/bin/env python3
"""
Migrate the shepherd facts table to add user_id column.

SQLite can't ALTER a PRIMARY KEY, so this recreates the table with
a composite PRIMARY KEY (user_id, key). Existing facts get user_id='local'.

Usage:
    python3 /tmp/migrate_facts_user_id.py [path_to_memory.db]

Default path: ~/.local/share/shepherd/memory.db
"""

import sqlite3
import sys
import os

def get_default_db_path():
    xdg = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
    return os.path.join(xdg, 'shepherd', 'memory.db')

def migrate(db_path):
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if migration is needed
    cursor.execute("PRAGMA table_info(facts)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'user_id' in columns:
        print("Facts table already has user_id column. Nothing to do.")
        conn.close()
        return

    print(f"Migrating {db_path}...")
    print(f"Current columns: {columns}")

    # Show existing facts
    cursor.execute("SELECT * FROM facts")
    rows = cursor.fetchall()
    print(f"Existing facts: {len(rows)}")
    for row in rows:
        print(f"  {row}")

    # Recreate table with composite primary key
    cursor.executescript("""
        BEGIN TRANSACTION;

        CREATE TABLE facts_new (
            user_id TEXT NOT NULL DEFAULT 'local',
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, key)
        );

        INSERT OR IGNORE INTO facts_new (user_id, key, value, created_at, updated_at)
            SELECT 'local', key, value, created_at, updated_at FROM facts;

        DROP TABLE facts;

        ALTER TABLE facts_new RENAME TO facts;

        COMMIT;
    """)

    # Verify
    cursor.execute("PRAGMA table_info(facts)")
    new_columns = [row[1] for row in cursor.fetchall()]
    print(f"New columns: {new_columns}")

    cursor.execute("SELECT * FROM facts")
    new_rows = cursor.fetchall()
    print(f"Migrated facts: {len(new_rows)}")
    for row in new_rows:
        print(f"  {row}")

    conn.close()
    print("Migration complete.")

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else get_default_db_path()
    migrate(db_path)
