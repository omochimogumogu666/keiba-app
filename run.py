"""
Run script for the JRA Horse Racing Prediction application.
"""
import os
from config.logging_config import setup_logging
from src.web.app import create_app

# Setup logging
setup_logging(log_level='INFO')

# Create Flask app
app = create_app()

if __name__ == '__main__':
    # Get environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'

    print(f"""
    ╔════════════════════════════════════════════════════╗
    ║   JRA Horse Racing Prediction Application         ║
    ╠════════════════════════════════════════════════════╣
    ║   Server: http://{host}:{port}                   ║
    ║   Environment: {'Development' if debug else 'Production'}                      ║
    ╚════════════════════════════════════════════════════╝
    """)

    app.run(host=host, port=port, debug=debug)
