"""
Declarative base for SQLAlchemy models.
This module is separate from database.py to avoid
requiring a database connection for metadata operations.
"""
from sqlalchemy.orm import declarative_base

Base = declarative_base()