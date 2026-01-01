"""
Database connection and session management.
"""
from contextlib import contextmanager
from src.data.models import db
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def init_db(app):
    """
    Initialize database with Flask app.

    Args:
        app: Flask application instance
    """
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully")


@contextmanager
def get_session():
    """
    Get database session context manager.

    Usage:
        with get_session() as session:
            session.query(Horse).all()
    """
    try:
        yield db.session
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.session.close()


def save_to_db(obj):
    """
    Save object to database.

    Args:
        obj: SQLAlchemy model instance

    Returns:
        Saved object
    """
    try:
        db.session.add(obj)
        db.session.commit()
        logger.debug(f"Saved to database: {obj}")
        return obj
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving to database: {e}")
        raise


def bulk_save_to_db(objects):
    """
    Bulk save objects to database.

    Args:
        objects: List of SQLAlchemy model instances

    Returns:
        Number of objects saved
    """
    try:
        db.session.bulk_save_objects(objects)
        db.session.commit()
        count = len(objects)
        logger.info(f"Bulk saved {count} objects to database")
        return count
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error bulk saving to database: {e}")
        raise


def delete_from_db(obj):
    """
    Delete object from database.

    Args:
        obj: SQLAlchemy model instance
    """
    try:
        db.session.delete(obj)
        db.session.commit()
        logger.debug(f"Deleted from database: {obj}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting from database: {e}")
        raise
