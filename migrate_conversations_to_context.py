#!/usr/bin/env python3
"""
One-time migration: rename 'conversations' table to 'context' (v2.32.0)

Usage:
  SQLite:     python3 migrate_conversations_to_context.py /path/to/memory.db
  PostgreSQL: python3 migrate_conversations_to_context.py "postgresql://user:pass@host:5432/dbname"

The default SQLite path is ~/.local/share/shepherd/memory.db
"""

import sys
import os


def migrate_sqlite(db_path):
    import sqlite3

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if old table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
    if not cursor.fetchone():
        print("No 'conversations' table found -- nothing to migrate (already done or fresh DB)")
        conn.close()
        return True

    # Check if new table already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context'")
    if cursor.fetchone():
        print("'context' table already exists -- migration already done")
        conn.close()
        return True

    print("Migrating: conversations -> context")

    # Drop old FTS table and triggers
    cursor.execute("DROP TRIGGER IF EXISTS conversations_ai")
    cursor.execute("DROP TRIGGER IF EXISTS conversations_ad")
    cursor.execute("DROP TRIGGER IF EXISTS conversations_au")
    cursor.execute("DROP TABLE IF EXISTS conversations_fts")

    # Rename the main table
    cursor.execute("ALTER TABLE conversations RENAME TO context")

    # Drop old index
    cursor.execute("DROP INDEX IF EXISTS idx_conversations_user_id")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_user_id ON context(user_id)")

    # Create new FTS table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(
            user_message,
            assistant_response,
            content='context',
            content_rowid='id'
        )
    """)

    # Create triggers
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS context_ai AFTER INSERT ON context BEGIN
            INSERT INTO context_fts(rowid, user_message, assistant_response)
            VALUES (new.id, new.user_message, new.assistant_response);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS context_ad AFTER DELETE ON context BEGIN
            INSERT INTO context_fts(context_fts, rowid, user_message, assistant_response)
            VALUES('delete', old.id, old.user_message, old.assistant_response);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS context_au AFTER UPDATE ON context BEGIN
            INSERT INTO context_fts(context_fts, rowid, user_message, assistant_response)
            VALUES('delete', old.id, old.user_message, old.assistant_response);
            INSERT INTO context_fts(rowid, user_message, assistant_response)
            VALUES (new.id, new.user_message, new.assistant_response);
        END
    """)

    # Rebuild FTS index from renamed table data
    cursor.execute("INSERT INTO context_fts(context_fts) VALUES('rebuild')")

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM context")
    count = cursor.fetchone()[0]
    print(f"Migration complete: {count} rows in 'context' table, FTS index rebuilt")

    conn.close()
    return True


def migrate_postgresql(connection_string):
    import psycopg2

    conn = psycopg2.connect(connection_string)
    conn.autocommit = True
    cursor = conn.cursor()

    # Check if old table exists
    cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name='conversations'")
    if not cursor.fetchone():
        print("No 'conversations' table found -- nothing to migrate (already done or fresh DB)")
        conn.close()
        return True

    # Check if new table already exists
    cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name='context'")
    if cursor.fetchone():
        print("'context' table already exists -- migration already done")
        conn.close()
        return True

    print("Migrating: conversations -> context")

    cursor.execute("ALTER TABLE conversations RENAME TO context")
    cursor.execute("ALTER INDEX IF EXISTS conversations_search_idx RENAME TO context_search_idx")
    cursor.execute("ALTER INDEX IF EXISTS conversations_timestamp_idx RENAME TO context_timestamp_idx")
    cursor.execute("ALTER INDEX IF EXISTS conversations_user_id_idx RENAME TO context_user_id_idx")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM context")
    count = cursor.fetchone()[0]
    print(f"Migration complete: {count} rows in 'context' table")

    conn.close()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default SQLite path
        default_path = os.path.expanduser("~/.local/share/shepherd/memory.db")
        print(f"No path specified, using default: {default_path}")
        sys.exit(0 if migrate_sqlite(default_path) else 1)

    target = sys.argv[1]

    if target.startswith("postgresql://") or target.startswith("postgres://"):
        sys.exit(0 if migrate_postgresql(target) else 1)
    else:
        sys.exit(0 if migrate_sqlite(target) else 1)
