"""
Search routes for the web application.
"""
from flask import Blueprint, render_template, request
from sqlalchemy import or_
from src.data.models import Horse, Jockey, Trainer, Race, Track
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

search_bp = Blueprint('search', __name__, url_prefix='/search')


@search_bp.route('/')
def search():
    """General search page."""
    query = request.args.get('q', '').strip()

    if not query:
        return render_template('search/index.html', query='', results=None)

    # Search across multiple entities
    results = {
        'horses': [],
        'jockeys': [],
        'trainers': [],
        'races': []
    }

    if len(query) >= 2:
        # Search horses
        horses = Horse.query.filter(
            Horse.name.contains(query)
        ).limit(10).all()
        results['horses'] = horses

        # Search jockeys
        jockeys = Jockey.query.filter(
            Jockey.name.contains(query)
        ).limit(10).all()
        results['jockeys'] = jockeys

        # Search trainers
        trainers = Trainer.query.filter(
            Trainer.name.contains(query)
        ).limit(10).all()
        results['trainers'] = trainers

        # Search races by name
        races = Race.query.filter(
            Race.race_name.contains(query)
        ).order_by(Race.race_date.desc()).limit(10).all()
        results['races'] = races

    total_results = (
        len(results['horses']) +
        len(results['jockeys']) +
        len(results['trainers']) +
        len(results['races'])
    )

    return render_template(
        'search/index.html',
        query=query,
        results=results,
        total_results=total_results
    )
