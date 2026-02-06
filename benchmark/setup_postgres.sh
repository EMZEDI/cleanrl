#!/bin/bash
# Setup PostgreSQL for Optuna on head node
# Run this ONCE before submitting tuning jobs

set -e

# Create PostgreSQL data directory
PGDATA="/scratch/shahradm/optuna_pg_data"
PGPORT=5432
PGHOST="localhost"
PGUSER="optuna"
PGPASSWORD="optuna_secure_pwd_2026"

echo "Setting up PostgreSQL for Optuna..."
echo "Data directory: $PGDATA"
echo "Port: $PGPORT"

# Load PostgreSQL module if available
module load postgresql || echo "PostgreSQL module not found, assuming system postgresql"

# Initialize database cluster
if [ ! -d "$PGDATA" ]; then
    echo "Initializing PostgreSQL cluster at $PGDATA..."
    initdb -D "$PGDATA" --auth=trust
else
    echo "PostgreSQL cluster already exists at $PGDATA"
fi

# Start PostgreSQL (if not already running)
echo "Starting PostgreSQL server..."
pg_ctl -D "$PGDATA" -l /scratch/shahradm/optuna_pg.log start 2>&1 | grep -v "already running" || true

# Wait for server to be ready
sleep 2

# Create optuna database and user
echo "Creating optuna user and database..."
psql -h localhost << EOSQL
CREATE USER IF NOT EXISTS optuna WITH PASSWORD 'optuna_secure_pwd_2026';
CREATE DATABASE IF NOT EXISTS optuna_humanoid OWNER optuna;
GRANT ALL PRIVILEGES ON DATABASE optuna_humanoid TO optuna;
\c optuna_humanoid
GRANT ALL ON SCHEMA public TO optuna;
EOSQL

if [ $? -ne 0 ]; then
    echo "Warning: User/database creation had issues, but this may be OK if they already exist."
fi

echo "PostgreSQL setup complete!"
echo ""
echo "Connection string for Optuna:"
echo "postgresql://optuna:optuna_secure_pwd_2026@$(hostname):5432/optuna_humanoid"
echo ""
echo "To stop PostgreSQL later:"
echo "  pg_ctl -D $PGDATA stop"
