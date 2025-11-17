from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from quiz_solver import QuizSolver

load_dotenv()

app = FastAPI()

# Your secret from the Google Form
MY_EMAIL = os.getenv("MY_EMAIL")
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/")
async def handle_quiz(request: QuizRequest):
    """
    Main endpoint to receive quiz tasks
    """
    # Verify secret
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    if request.email != MY_EMAIL:
        raise HTTPException(status_code=403, detail="Invalid email")
    
    # Start quiz solving in background
    solver = QuizSolver(request.email, request.secret)
    
    # Run asynchronously to not block the response
    asyncio.create_task(solver.solve_quiz_chain(request.url))
    
    return {
        "status": "accepted",
        "message": "Quiz solving started"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
