import sqlite3
import numpy as np

DB_PATH = "face.db"
START_ID = 101

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()

def insert_embedding(embedding: np.ndarray) -> int:
    data = np.asarray(embedding, dtype=np.float32).tobytes()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), ?) AS max_id FROM face_embeddings",
            (START_ID - 1,)
        ).fetchone()

        new_id = int(row["max_id"]) + 1

        conn.execute(
            "INSERT INTO face_embeddings (id, embedding) VALUES (?, ?)",
            (new_id, data)
        )
        conn.commit()

    return new_id

def load_all_embeddings():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM face_embeddings ORDER BY id"
        ).fetchall()

    result = []
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        result.append((int(row["id"]), embedding))

    return result

def find_best_match(input_embedding: np.ndarray, threshold: float = 0.50):
    records = load_all_embeddings()

    if not records:
        return {
            "found": False,
            "id": None,
            "similarity": None,
            "message": "Database masih kosong."
        }

    input_embedding = np.asarray(input_embedding, dtype=np.float32)
    input_norm = np.linalg.norm(input_embedding)

    if input_norm == 0:
        return {
            "found": False,
            "id": None,
            "similarity": None,
            "message": "Embedding input tidak valid."
        }

    input_embedding = input_embedding / input_norm

    best_id = None
    best_similarity = -1.0

    for record_id, db_embedding in records:
        db_norm = np.linalg.norm(db_embedding)
        if db_norm == 0:
            continue

        db_embedding = db_embedding / db_norm
        similarity = float(np.dot(input_embedding, db_embedding))

        if similarity > best_similarity:
            best_similarity = similarity
            best_id = record_id

    found = best_similarity >= threshold

    return {
        "found": found,
        "id": best_id if found else None,
        "similarity": round(best_similarity, 6),
        "threshold": threshold
    }

def count_records() -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM face_embeddings"
        ).fetchone()
    return int(row["total"])
