"""Add public-holiday cache and optional SMS alert contact details.

Revision ID: 003_public_holidays_and_alert_contacts
Revises: a0a643db4f4b
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "003_public_holidays_and_alert_contacts"
down_revision = "a0a643db4f4b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_users", sa.Column("phone_number", sa.String(), nullable=True))
    op.create_table(
        "public_holidays",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False),
        sa.Column("holiday_date", sa.Date(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("local_name", sa.String(), nullable=False),
        sa.Column("global_holiday", sa.Boolean(), nullable=False),
        sa.Column("types", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("country_code", "holiday_date", "name", name="public_holidays_country_date_name_key"),
    )
    op.create_index("ix_public_holidays_country_code", "public_holidays", ["country_code"])
    op.create_index("ix_public_holidays_holiday_date", "public_holidays", ["holiday_date"])
    op.create_index("ix_public_holidays_year", "public_holidays", ["year"])


def downgrade() -> None:
    op.drop_index("ix_public_holidays_year", table_name="public_holidays")
    op.drop_index("ix_public_holidays_holiday_date", table_name="public_holidays")
    op.drop_index("ix_public_holidays_country_code", table_name="public_holidays")
    op.drop_table("public_holidays")
    op.drop_column("app_users", "phone_number")
