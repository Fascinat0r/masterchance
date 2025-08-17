"""add exam_sessions

Revision ID: 9a1b2c3d4e5f
Revises: 8b00048dd7d5
Create Date: 2025-08-15 23:59:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9a1b2c3d4e5f'
down_revision: Union[str, Sequence[str], None] = '8b00048dd7d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'exam_sessions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('program_code', sa.String(), sa.ForeignKey('programs.code'), nullable=False),
        sa.Column('exam_code', sa.String(), nullable=True),
        sa.Column('dt', sa.DateTime(), nullable=False),
        sa.Column('institute', sa.String(), nullable=True),
        sa.Column('education_form', sa.String(), nullable=True),
        sa.Column('contract', sa.String(), nullable=True),
        sa.Column('program_name', sa.String(), nullable=True),
        sa.Column('program_pdf_url', sa.String(), nullable=True),
    )
    op.create_index('ix_exam_sessions_program_code', 'exam_sessions', ['program_code'])
    op.create_index('ix_exam_sessions_program_dt', 'exam_sessions', ['program_code', 'dt'])


def downgrade() -> None:
    op.drop_index('ix_exam_sessions_program_dt', table_name='exam_sessions')
    op.drop_index('ix_exam_sessions_program_code', table_name='exam_sessions')
    op.drop_table('exam_sessions')
