"""create document segments table

Revision ID: a7c4e91d2026
Revises: d16a20261022
Create Date: 2026-10-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7c4e91d2026"
down_revision: Union[str, None] = "d16a20261022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_segments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("start_seconds", sa.Float(), nullable=True),
        sa.Column("end_seconds", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "ordinal >= 0",
            name="ck_document_segments_ordinal_nonnegative",
        ),
        sa.CheckConstraint(
            "text_content ~ '[^[:space:]]'",
            name="ck_document_segments_text_nonempty",
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number > 0",
            name="ck_document_segments_page_number_positive",
        ),
        sa.CheckConstraint(
            "(start_seconds IS NULL AND end_seconds IS NULL) OR "
            "(start_seconds IS NOT NULL AND end_seconds IS NOT NULL AND "
            "start_seconds >= 0 AND end_seconds >= start_seconds)",
            name="ck_document_segments_timestamps_valid",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_segments_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "ordinal",
            name="uq_document_segments_document_id_ordinal",
        ),
    )


def downgrade() -> None:
    op.drop_table("document_segments")
