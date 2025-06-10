import json
from fastapi import FastAPI, HTTPException, Depends
import joblib
from pydantic import BaseModel, ConfigDict, Field, validator
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from torch import cosine_similarity
from gemini_fallback import analyze_stack_trace, ask_gemini_exception_type
from log_recommendation import LogRecommendationSystem
from database_models import get_db, SessionLocal, Log, Ticket, Solution
from sqlalchemy.orm import Session

class ModelChoice(str, Enum):
    openai = "openai"
    gemini = "gemini"

class LogRequest(BaseModel):
    message: str
    model: ModelChoice = ModelChoice.openai

class LogResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    exception_type: str
    score: float
    matchedFrom: str
    model_used: str

class AnalysisRequest(BaseModel):
    message: str

class AnalysisResponse(BaseModel):
    exception_type: str
    stack_trace: str | None
    analysis: dict | None
    error: str | None

class RecommendationRequest(BaseModel):
    log_message: str = Field(..., min_length=10, description="The error log message to find recommendations for")
    k: int = Field(5, ge=1, le=20, description="Number of recommendations to return (1-20)")
    
    @validator('log_message')
    def validate_log_message(cls, v):
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Log message must be at least 10 characters long")
        return v

class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    similar_logs: List[Dict] = []
    message: str = ""
    query_analysis: Optional[Dict] = None

app = FastAPI()

# Load models
try:
    vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
    df = joblib.load("model/messages_df.joblib")
    X = vectorizer.transform(df['message'])
except Exception as e:
    print(f"Warning: Could not load model files: {str(e)}")
    vectorizer = None
    df = None
    X = None

# Initialize the recommendation system
recommender = LogRecommendationSystem()

# Initialize the index from database on startup
@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        recommender.build_index_from_db(db)
    finally:
        db.close()

@app.get("/check-database")
def check_database(db: Session = Depends(get_db)):
    """Check the state of database relationships"""
    try:
        # Check logs
        logs = db.query(Log).all()
        log_info = [{"id": log.id, "type": log.type, "description": log.description} for log in logs]
        
        # Check tickets
        tickets = db.query(Ticket).all()
        ticket_info = [{"id": ticket.id, "log_id": ticket.log_id, "status": ticket.status} for ticket in tickets]
        
        # Check solutions
        solutions = db.query(Solution).all()
        solution_info = [{"id": sol.id, "ticket_id": sol.ticket_id, "content": sol.content} for sol in solutions]
        
        return {
            "total_logs": len(logs),
            "total_tickets": len(tickets),
            "total_solutions": len(solutions),
            "logs": log_info,
            "tickets": ticket_info,
            "solutions": solution_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-log", response_model=LogResponse)
def classify_log(request: LogRequest):
    try:
        if vectorizer and df is not None and X is not None:
            new_vec = vectorizer.transform([request.message])
            similarities = cosine_similarity(new_vec, X)
            best_match_index = similarities.argmax()
            best_score = float(similarities[0][best_match_index])

            if best_score > 0.7:
                return {
                    "exception_type": df.iloc[best_match_index]['category'],
                    "score": best_score,
                    "matchedFrom": "tfidf",
                    "model_used": "none"
                }
        
        # Use Gemini model
        exception_type = ask_gemini_exception_type(request.message)
        model_used = "gemini"
            
        return {
            "exception_type": exception_type,
            "score": 0.0,
            "matchedFrom": "ai",
            "model_used": model_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-stack-trace", response_model=AnalysisResponse)
async def analyze_log(request: AnalysisRequest):
    """Analyze a log message to extract and analyze its stack trace."""
    try:
        # Get exception type
        exception_type = "Unknown"
        try:
            exception_type = ask_gemini_exception_type(request.message)
        except Exception as e:
            logger.error(f"Error getting exception type: {str(e)}")
            exception_type = "Error determining exception type"
        
        # Get comprehensive analysis
        try:
            analysis_result = analyze_stack_trace(request.message)
        except Exception as e:
            logger.error(f"Error analyzing stack trace: {str(e)}")
            return {
                "exception_type": exception_type,
                "stack_trace": None,
                "analysis": None,
                "error": f"Failed to analyze stack trace: {str(e)}"
            }
        
        # Parse the JSON string if it's a string
        if isinstance(analysis_result.get("analysis", ""), str):
            try:
                # Remove markdown code block markers if present
                analysis_text = analysis_result["analysis"].replace("```json", "").replace("```", "").strip()
                analysis_dict = json.loads(analysis_text)
                analysis_result["analysis"] = analysis_dict
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse analysis JSON: {str(e)}")
                analysis_result["analysis"] = {"error": "Could not parse JSON analysis", "raw_text": analysis_result["analysis"]}
        
        return {
            "exception_type": exception_type,
            "stack_trace": analysis_result.get("stack_trace"),
            "analysis": analysis_result.get("analysis"),
            "error": analysis_result.get("error")
        }
    except Exception as e:
        logger.error(f"Unhandled error in analyze_log: {str(e)}")
        return {
            "exception_type": "Error",
            "stack_trace": None,
            "analysis": None,
            "error": f"Failed to process request: {str(e)}"
        }

@app.post("/recommend-solutions", response_model=RecommendationResponse)
def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get solution recommendations for a given log message."""
    try:
        # Check if the input appears to be an error log
        if not any(err_term in request.log_message.lower() for err_term in ['error', 'exception', 'failed', 'failure']):
            # Do a more thorough check through the recommendation system
            result = recommender.find_similar_logs(request.log_message, k=request.k)
            
            # Check if the recommendation system identified it as not a log
            if "not appear to be an error log" in result.get("message", ""):
                result["similar_logs"] = []
                result["message"] = "The input does not appear to be an error log. Please enter a valid error message or stack trace."
                return result
                
        # Process normally if it appears to be an error log
        result = recommender.find_similar_logs(request.log_message, k=request.k)
        
        # Check if we have results and they meet the minimum threshold
        if not result["similar_logs"] and "low confidence" not in result.get("message", ""):
            result["message"] = "No matching error logs found in our database. Try providing a more complete error message."
            
        return result
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-recommendations")
def debug_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Debug mode for solution recommendations with detailed matching analysis."""
    try:
        result = recommender.debug_similar_logs(request.log_message, k=request.k)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/system-info")
def get_system_info():
    """Get information about the recommendation system state."""
    try:
        return recommender.get_system_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-thresholds")
def update_thresholds(
    similarity: Optional[float] = None,
    term_ratio: Optional[float] = None,
    debug_mode: Optional[bool] = None,
    min_fallback_similarity: Optional[float] = None
):
    """Update recommendation system thresholds."""
    try:
        response = {"message": "Updated thresholds: "}
        if similarity is not None:
            recommender.set_similarity_threshold(similarity)
            response["similarity"] = similarity
        
        if term_ratio is not None:
            recommender.set_term_ratio_threshold(term_ratio)
            response["term_ratio"] = term_ratio
            
        if debug_mode is not None:
            recommender.set_debug_mode(debug_mode)
            response["debug_mode"] = debug_mode
        
        if min_fallback_similarity is not None:
            if 0.0 <= min_fallback_similarity <= 1.0:
                # Use the dedicated method instead of direct assignment
                recommender.set_fallback_similarity_threshold(min_fallback_similarity)
                response["min_fallback_similarity"] = min_fallback_similarity
            else:
                raise ValueError(f"Invalid min_fallback_similarity value: {min_fallback_similarity}. Must be between 0.0 and 1.0")
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
