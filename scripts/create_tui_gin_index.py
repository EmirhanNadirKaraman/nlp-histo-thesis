from sqlalchemy import text
from nlp_histo.database import get_db_connection


def create_index():
    db = get_db_connection()
    # Use the underlying engine to execute raw SQL
    with db.engine.connect() as connection:
        print("Creating GIN index on semantic_types... this may take a minute.")
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_entity_semantic_types ON entities USING GIN (semantic_types);")
        )
        connection.commit()
        print("✓ Index created successfully!")


if __name__ == "__main__":
    create_index()
