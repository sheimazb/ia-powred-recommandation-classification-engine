from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from openai_fallback import ask_openai_exception_type
from gemini_fallback import ask_gemini_exception_type, analyze_stack_trace
from enum import Enum
import json
import uvicorn

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
        exception_type = ask_gemini_exception_type(request.message)
        
        # Get comprehensive analysis
        analysis_result = analyze_stack_trace(request.message)
        
        # Parse the JSON string if it's a string
        if isinstance(analysis_result["analysis"], str):
            try:
                # Remove markdown code block markers if present
                analysis_text = analysis_result["analysis"].replace("```json", "").replace("```", "").strip()
                analysis_dict = json.loads(analysis_text)
                analysis_result["analysis"] = analysis_dict
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse analysis JSON: {str(e)}")
        
        return {
            "exception_type": exception_type,
            "stack_trace": analysis_result["stack_trace"],
            "analysis": analysis_result["analysis"],
            "error": analysis_result["error"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
