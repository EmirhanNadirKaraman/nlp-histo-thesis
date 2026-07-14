"""
Database connection utilities for PostgreSQL.
Handles connection pooling and session management.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from .models import Base

# Load .env file automatically when this module is imported
# This way, all scripts that use the database don't need to load it themselves
try:
    from dotenv import load_dotenv
    # Look for .env file in project root (parent of database folder)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed - will use environment variables or defaults
    pass

# Database configuration
# These can be overridden by environment variables (loaded from .env above)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'nlp_histo'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
}


def get_database_url(config=None):
    """
    Construct PostgreSQL database URL from configuration.

    Args:
        config: Optional dictionary with database configuration.
                If None, uses DB_CONFIG.

    Returns:
        str: PostgreSQL connection URL
    """
    if config is None:
        config = DB_CONFIG
    
    return (
        f"postgresql://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['database']}"
    )


class DatabaseConnection:
    """
    Manages database connection and session lifecycle.
    """

    def __init__(self, database_url=None, echo=False):
        """
        Initialize database connection.

        Args:
            database_url: PostgreSQL connection URL. If None, constructs from DB_CONFIG.
            echo: If True, SQL statements will be logged (useful for debugging).
        """
        if database_url is None:
            database_url = get_database_url()

        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=15,         # Increased from 10 for parallel processing
            max_overflow=25       # Increased from 20 (total capacity: 40)
        )

        # Create a configured "Session" class
        session_factory = sessionmaker(bind=self.engine)

        # Create a thread-local Session
        self.Session = scoped_session(session_factory)

    def create_tables(self):
        """
        Create all tables defined in the models.
        """
        Base.metadata.create_all(self.engine)
        print("Database tables created successfully!")

    def drop_tables(self):
        """
        Drop all tables defined in the models.
        WARNING: This will delete all data!
        """
        Base.metadata.drop_all(self.engine)
        print("Database tables dropped successfully!")

    def get_session(self):
        """
        Get a new database session.

        Returns:
            Session: SQLAlchemy session object
        """
        return self.Session()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session_scope() as session:
                session.add(new_document)
                # ... more operations ...
            # Automatically commits or rolls back
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def close(self):
        """
        Close all sessions and dispose of the connection pool.
        """
        self.Session.remove()
        self.engine.dispose()


# Global database connection instance
_db_instance = None


def get_db_connection(database_url=None, echo=False):
    """
    Get or create a global database connection instance.

    Args:
        database_url: PostgreSQL connection URL
        echo: If True, SQL statements will be logged

    Returns:
        DatabaseConnection: Database connection instance
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = DatabaseConnection(database_url=database_url, echo=echo)

    return _db_instance


def close_db_connection():
    """
    Close the global database connection instance.
    """
    global _db_instance

    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None
