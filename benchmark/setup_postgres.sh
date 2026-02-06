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

# Start PostgreSQL
echo "Starting PostgreSQL server..."
pg_ctl -D "$PGDATA" -l /scratch/shahradm/optuna_pg.log start

# Wait for server to be ready
sleep 3

# Create optuna database and user
echo "Creating optuna user and database..."
createdb -U postgres optuna_humanoid 2>/dev/null || echo "Database optuna_humanoid already exists"
createuser -U postgres optuna 2>/dev/null || echo "User optuna already exists"

# Grant privileges
psql -U postgres -d optuna_humanoid << EOF
ALTER USER optuna WITH PASSWORD 'optuna_secure_pwd_2026';
GRANT ALL PRIVILEGES ON DATABASE optuna_humanoid TO optuna;
EOF

echo "PostgreSQL setup complete!"
echo ""
echo "Connection string for Optuna:"
echo "postgresql://optuna:optuna_secure_pwd_2026@$(hostname):5432/optuna_humanoid"
echo ""
echo "To stop PostgreSQL later:"
echo "  pg_ctl -D $PGDATA stop"
