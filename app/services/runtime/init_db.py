from __future__ import annotations

from backend.app.database import database_status, ensure_database
from backend.app.services.auth import bootstrap_local_root_user


def main() -> None:
    print(database_status())
    if not ensure_database(strict=True):
        raise SystemExit("Database was not initialized. Set DATABASE_URL and install SQLAlchemy/psycopg.")
    bootstrap_local_root_user()
    print("Database schema is ready and the configured root user has been bootstrapped.")


if __name__ == "__main__":
    main()
