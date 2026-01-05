#!/bin/bash
# Docker environment database migration script

set -e

echo "Starting database migration in Docker environment..."

# Wait for database to be ready
echo "Waiting for PostgreSQL to be ready..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' 2>/dev/null; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "PostgreSQL is up - executing migrations"

# Set Flask app
export FLASK_APP=src.web.app:create_app

# Run database migrations
echo "Running database migrations..."
flask db upgrade

echo "Database migration completed successfully!"
