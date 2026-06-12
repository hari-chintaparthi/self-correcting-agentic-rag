from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
from pydantic import BaseModel
from rag_pipeline import app as rag_graph, ingest_document


app = FastAPI(
    title="Self-Correcting Agentic RAG Platform",
    description="FastAPI gateway that utilizes a LangGraph state machine to perform self-correcting semantic retrieval and hallucination filtering.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    generation: str
    rewritten_query: str
    relevance_score: float
    hallucination_grade: str
    loop_count: int
    retrieved_documents: list[str]

@app.post("/query", response_model=QueryResponse)
async def execute_rag_pipeline(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        # Initial state variables
        initial_state = {
            "query": request.query,
            "rewritten_query": "",
            "retrieved_documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "hallucination_grade": "NO",
            "loop_count": 0
        }
        
        # Execute state machine graph
        final_state = rag_graph.invoke(initial_state)
        
        return QueryResponse(
            query=final_state["query"],
            generation=final_state["generation"],
            rewritten_query=final_state.get("rewritten_query", ""),
            relevance_score=final_state.get("relevance_score", 0.0),
            hallucination_grade=final_state.get("hallucination_grade", "NO"),
            loop_count=final_state.get("loop_count", 0),
            retrieved_documents=final_state.get("retrieved_documents", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Create a temp directory inside the project workspace
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    try:
        # Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger ingestion
        message = ingest_document(file_path, file.filename)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host=host, port=port, reload=True)

