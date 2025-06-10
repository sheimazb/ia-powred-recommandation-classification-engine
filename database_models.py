from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get PostgreSQL connection details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0000")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tickets")

# Create PostgreSQL URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger.info(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME}")

Base = declarative_base()

class Log(Base):
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True)
    analysis_ia = Column(String)
    class_name = Column(String)
    container_id = Column(String)
    container_name = Column(String)
    created_at = Column(DateTime)
    custom_message = Column(String)
    description = Column(String)
    error_code = Column(String)
    exception_type = Column(String)
    original_timestamp = Column(DateTime)
    pid = Column(String)
    project_id = Column(String)
    severity = Column(String)
    source = Column(String)
    stack_trace = Column(String)
    tag = Column(String)
    tenant = Column(String)
    thread = Column(String)
    timestamp = Column(DateTime)
    type = Column(String)
    
    # Relationship
    ticket = relationship("Ticket", back_populates="log", uselist=False)

class Ticket(Base):
    __tablename__ = 'tickets'
    
    id = Column(Integer, primary_key=True)
    assigned_to_user_id = Column(String)
    created_at = Column(DateTime)
    creator_user_id = Column(String)
    description = Column(String)
    priority = Column(String)
    status = Column(String)
    tenant = Column(String)
    title = Column(String)
    updated_at = Column(DateTime)
    user_email = Column(String)
    log_id = Column(Integer, ForeignKey('logs.id'))
    
    # Relationships
    log = relationship("Log", back_populates="ticket")
    solution = relationship("Solution", back_populates="ticket", uselist=False)

class Solution(Base):
    __tablename__ = 'solutions'
    
    id = Column(Integer, primary_key=True)
    author_user_id = Column(String)
    category = Column(String)
    complexity = Column(String)
    content = Column(String)
    cost_estimation = Column(String)
    created_at = Column(DateTime)
    estimated_time = Column(String)
    status = Column(String)
    tenant = Column(String)
    title = Column(String)
    updated_at = Column(DateTime)
    ticket_id = Column(Integer, ForeignKey('tickets.id'))
    
    # Relationship
    ticket = relationship("Ticket", back_populates="solution")

try:
    # Create engine with connection debugging
    engine = create_engine(DATABASE_URL, echo=True)
    
    # Test the connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    raise

# SessionLocal class
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        # Test the connection with proper text() wrapper
        db.execute(text("SELECT 1"))
        logger.info("Database session created successfully")
        yield db
    except Exception as e:
        logger.error(f"Error with database session: {str(e)}")
        raise
    finally:
        logger.info("Closing database session")
        db.close() 