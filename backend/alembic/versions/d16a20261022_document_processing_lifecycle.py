"""add document processing lifecycle status and safe error code

Revision ID: d16a20261022
Revises: c5ede9494309
Create Date: 2026-10-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d16a20261022"
down_revision: Union[str, None] = "c5ede9494309"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("processing_error_code", sa.String(length=50), nullable=True),
    )
    op.drop_constraint(
        "ck_documents_processing_status",
        "documents",
        type_="check",
    )
    op.create_check_constraint(
        "ck_documents_processing_status",
        "documents",
        "processing_status IN "
        "('pending', 'uploaded', 'processing', 'completed', 'failed')",
    )
    op.create_check_constraint(
        "ck_documents_processing_error_code",
        "documents",
        "processing_error_code IS NULL OR "
        "(processing_status = 'failed' AND "
        "processing_error_code = 'processing_failed')",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE documents SET processing_status = 'pending' "
        "WHERE processing_status = 'uploaded'"
    )
    op.drop_constraint(
        "ck_documents_processing_error_code",
        "documents",
        type_="check",
    )
    op.drop_column("documents", "processing_error_code")
    op.drop_constraint(
        "ck_documents_processing_status",
        "documents",
        type_="check",
    )
    op.create_check_constraint(
        "ck_documents_processing_status",
        "documents",
        "processing_status IN ('pending', 'processing', 'completed', 'failed')",
    )
