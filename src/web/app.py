"""
Flask application factory.
"""
from flask import Flask, render_template
from config.settings import get_config
from config.logging_config import setup_logging
from src.data.models import db
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def create_app(config_name=None):
    """
    Create and configure Flask application.

    Args:
        config_name: Configuration name (development, production, testing)

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    config.init_app(app)

    # Setup logging
    setup_logging(
        log_file=str(config.LOG_FILE),
        log_level=config.LOG_LEVEL
    )

    # Initialize database
    db.init_app(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Create database tables
    with app.app_context():
        db.create_all()

    logger.info(f"Flask application created with {config_name or 'default'} config")

    return app


def register_blueprints(app):
    """Register Flask blueprints."""
    from src.web.routes.main import main_bp
    from src.web.routes.predictions import predictions_bp
    from src.web.routes.entities import entities_bp
    from src.web.routes.search import search_bp
    from src.web.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(entities_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(api_bp)

    logger.info("Blueprints registered")


def register_error_handlers(app):
    """Register error handlers."""

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500

    logger.info("Error handlers registered")


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
