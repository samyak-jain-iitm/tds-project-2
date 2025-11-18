from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import os
from quiz_solver import QuizSolver

app = FastAPI(title="LLM Quiz Solver")

MY_EMAIL = os.getenv("MY_EMAIL")
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "LLM Quiz Solver API",
        "email_configured": MY_EMAIL is not None,
        "secret_configured": MY_SECRET is not None
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy"}

@app.post("/")
async def handle_quiz(request: QuizRequest):
    """
    Main endpoint to receive quiz tasks
    Validates credentials and starts quiz solving
    """
    if not MY_SECRET:
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: SECRET not set"
        )
    
    if request.secret != MY_SECRET:
        raise HTTPException(
            status_code=403, 
            detail="Invalid secret"
        )
    
    if not MY_EMAIL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: EMAIL not set"
        )
    
    if request.email != MY_EMAIL:
        raise HTTPException(
            status_code=403,
            detail="Invalid email"
        )
    
    solver = QuizSolver(request.email, request.secret)
    asyncio.create_task(solver.solve_quiz_chain(request.url))
    
    return {
        "status": "accepted",
        "message": "Quiz solving started",
        "url": request.url
    }

@app.post("/test")
async def test_endpoint(request: QuizRequest):
    """Test endpoint for validation"""
    return {
        "status": "test_successful",
        "email_match": request.email == MY_EMAIL,
        "secret_match": request.secret == MY_SECRET,
        "url_received": request.url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
