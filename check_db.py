from database_models import SessionLocal, Log, Ticket, Solution
from sqlalchemy import func
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database():
    db = SessionLocal()
    try:
        # Check counts for each table
        log_count = db.query(func.count(Log.id)).scalar()
        ticket_count = db.query(func.count(Ticket.id)).scalar()
        solution_count = db.query(func.count(Solution.id)).scalar()
        
        logger.info(f"Database Statistics:")
        logger.info(f"Number of Logs: {log_count}")
        logger.info(f"Number of Tickets: {ticket_count}")
        logger.info(f"Number of Solutions: {solution_count}")
        
        # Sample data from each table
        logger.info("\nSample Log entries:")
        sample_logs = db.query(Log).limit(3).all()
        for log in sample_logs:
            logger.info(f"Log ID: {log.id}")
            logger.info(f"Error Type: {log.type}")
            logger.info(f"Description: {log.description}")
            logger.info("---")
        
        logger.info("\nSample Ticket entries:")
        sample_tickets = db.query(Ticket).limit(3).all()
        for ticket in sample_tickets:
            logger.info(f"Ticket ID: {ticket.id}")
            logger.info(f"Title: {ticket.title}")
            logger.info(f"Status: {ticket.status}")
            logger.info("---")
        
        logger.info("\nSample Solution entries:")
        sample_solutions = db.query(Solution).limit(3).all()
        for solution in sample_solutions:
            logger.info(f"Solution ID: {solution.id}")
            logger.info(f"Title: {solution.title}")
            logger.info(f"Content: {solution.content[:100]}...")  # First 100 chars
            logger.info("---")

    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    check_database() 