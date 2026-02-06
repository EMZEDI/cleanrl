#!/bin/bash
# Setup PostgreSQL for Optuna on head node
# Run this ONCE before submitting tuning jobs

set -e

# Create PostgreSQL data directory
# PASSWD: optuna_secure_pwd_2026
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
STATUS=$(pg_ctl -D "$PGDATA" status 2>&1 || true)
if echo "$STATUS" | grep -q "server is running"; then
    echo "PostgreSQL is already running (PID: $(echo $STATUS | grep -oP 'PID: \K[0-9]+'))"
else
    pg_ctl -D "$PGDATA" -l /scratch/shahradm/optuna_pg.log start 2>&1 | grep -v "already running" || true
fi

# Wait for server to be ready
sleep 2

# Create optuna database and user (connect to template1 which always exists)
echo "Creating optuna user and database..."
psql -h localhost -d template1 << EOSQL
CREATE USER IF NOT EXISTS optuna WITH PASSWORD 'optuna_secure_pwd_2026';
CREATE DATABASE IF NOT EXISTS optuna_humanoid OWNER optuna;
GRANT ALL PRIVILEGES ON DATABASE optuna_humanoid TO optuna;
EOSQL

# Grant schema permissions
psql -h localhost -d optuna_humanoid << EOSQL
GRANT ALL ON SCHEMA public TO optuna;
EOSQL

echo "PostgreSQL setup complete!"
echo ""
echo "Connection string for Optuna:"
echo "postgresql://optuna:optuna_secure_pwd_2026@$(hostname):5432/optuna_humanoid"
echo ""
echo "To stop PostgreSQL later:"
echo "  pg_ctl -D $PGDATA stop"
