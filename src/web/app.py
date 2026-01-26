"""
Flask application factory.

このモジュールはFlaskアプリケーションの初期化と設定を担当します。
- CSRF保護（Flask-WTF）
- APIレート制限（Flask-Limiter）
- データベース初期化
- キャッシュ設定
- ブループリント登録
"""
from flask import Flask, render_template
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config.settings import get_config
from config.logging_config import setup_logging
from src.data.models import db
from src.web.cache import init_cache
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)
migrate = Migrate()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)


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

    # Initialize Flask-Migrate
    migrate.init_app(app, db)

    # Initialize cache
    init_cache(app)

    # Initialize CSRF protection
    csrf.init_app(app)

    # Initialize rate limiter
    limiter.init_app(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Register before_request hook for auto status update
    register_status_update_hook(app)

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
    from src.web.routes.simulation import simulation_bp
    from src.web.routes.scraping import scraping_bp
    from src.web.routes.analysis import analysis_bp
    from src.web.routes.training import training_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(entities_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(simulation_bp)
    app.register_blueprint(scraping_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(training_bp)

    # CSRF exemption for API endpoints (they use JSON)
    csrf.exempt(scraping_bp)
    csrf.exempt(predictions_bp)
    csrf.exempt(analysis_bp)
    csrf.exempt(training_bp)

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


def register_status_update_hook(app):
    """
    レースステータス自動更新のbefore_requestフックを登録。

    静的ファイルやAPIエンドポイント以外のページアクセス時に、
    レースのステータスを自動更新する。60秒間隔で実行。
    """
    from flask import request
    from src.data.database import auto_update_race_status

    @app.before_request
    def update_race_status():
        # 静的ファイルへのリクエストはスキップ
        if request.path.startswith('/static/'):
            return

        # SSE（Server-Sent Events）エンドポイントはスキップ
        if '/api/progress/' in request.path:
            return

        # ステータスを自動更新（60秒間隔チェック内蔵）
        try:
            auto_update_race_status()
        except Exception as e:
            # ステータス更新の失敗はリクエスト処理を妨げない
            logger.warning(f"レースステータス自動更新でエラー: {e}")

    logger.info("Race status auto-update hook registered")


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5001)
