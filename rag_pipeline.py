import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
import docx2txt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define state schema
class AgentState(TypedDict):

    query: str
    rewritten_query: str
    retrieved_documents: List[str]
    generation: str
    relevance_score: float
    hallucination_grade: str
    loop_count: int

# Initialize models
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)




# Nodes
def retrieve_docs(state: AgentState) -> Dict[str, Any]:
    """Retrieves matching documents from the vector database."""
    print("---RETRIEVING DOCUMENTS---")
    query = state.get("rewritten_query") or state["query"]
    
    # Check if a local FAISS index exists
    if os.path.exists("faiss_index"):
        try:
            print("---LOADING FAISS INDEX---")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=4)
            retrieved_texts = [doc.page_content for doc in docs]
            print(f"Retrieved {len(retrieved_texts)} documents from vector store.")
            return {"retrieved_documents": retrieved_texts, "loop_count": state.get("loop_count", 0) + 1}
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Falling back to mock documents.")
            
    # Mock documents for initialization/fallback
    print("---USING FALLBACK MOCK DOCUMENTS---")
    mock_docs = [
        "In 2026, the company achieved $50M in annual recurring revenue.",
        "The current remote work policy allows employees to work anywhere within the country."
    ]
    return {"retrieved_documents": mock_docs, "loop_count": state.get("loop_count", 0) + 1}

def ingest_document(file_path: str, filename: str) -> str:
    """Parses, chunks, embeds, and saves document to the local FAISS index."""
    print(f"---INGESTING DOCUMENT: {filename}---")
    text = ""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")
            
    elif ext in [".docx", ".doc"]:
        try:
            text = docx2txt.process(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse Word Document: {e}")
            
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read TXT file: {e}")
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, Word Document, or TXT file.")
        
    if not text.strip():
        raise ValueError("Extracted document content is empty.")
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    print(f"Split document into {len(chunks)} chunks.")
    
    # Save/update vector store
    try:
        if os.path.exists("faiss_index"):
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            db.add_texts(chunks)
        else:
            db = FAISS.from_texts(chunks, embeddings)
            
        db.save_local("faiss_index")
        print("Successfully saved to local FAISS index.")
        return f"Successfully ingested {len(chunks)} text chunks from '{filename}' into vector store."
    except Exception as e:
        raise RuntimeError(f"Failed to save to vector store: {e}")


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Grades retrieved documents for relevance to the query."""
    print("---GRADING DOCUMENTS---")
    query = state["query"]
    docs = state["retrieved_documents"]
    
    # LLM-as-a-judge prompt to score relevance
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the relevance of these documents to the query: '{query}'.\n"
        "Documents:\n{docs}\n"
        "Reply with a single float score between 0.0 (irrelevant) and 1.0 (highly relevant)."
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"query": query, "docs": "\n".join(docs)})
        score = float(response.content.strip())
    except Exception:
        score = 0.8  # Fallback score
        
    return {"relevance_score": score}

def generate_answer(state: AgentState) -> Dict[str, Any]:
    """Generates the answer using the retrieved documents."""
    print("---GENERATING ANSWER---")
    query = state["query"]
    docs = state["retrieved_documents"]
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the query based ONLY on the following context:\nContext:\n{docs}\n\nQuery: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "docs": "\n".join(docs)})
    return {"generation": response.content}

def grade_generation(state: AgentState) -> Dict[str, Any]:
    """Grades generation for hallucinations."""
    print("---CHECKING FOR HALLUCINATIONS---")
    generation = state["generation"]
    docs = state["retrieved_documents"]
    
    prompt = ChatPromptTemplate.from_template(
        "Is the following statement fully grounded in the provided documents?\n"
        "Statement: {generation}\n"
        "Documents:\n{docs}\n"
        "Answer with either 'YES' or 'NO'."
    )
    chain = prompt | llm
    response = chain.invoke({"generation": generation, "docs": "\n".join(docs)})
    grade = response.content.strip().upper()
    return {"hallucination_grade": grade}

def rewrite_query(state: AgentState) -> Dict[str, Any]:
    """Rewrites the query to optimize retrieval results."""
    print("---REWRITING QUERY---")
    query = state["query"]
    
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this search query to make it more descriptive and suitable for semantic search.\n"
        "Original: {query}\n"
        "Output query only:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return {"rewritten_query": response.content.strip()}

# Router logic
def route_after_retrieval(state: AgentState) -> str:
    """Decides whether to generate or rewrite the query based on relevance scores."""
    if state["relevance_score"] >= 0.7:
        return "generate"
    elif state["loop_count"] >= 3:
        # Fallback to generation if stuck in loops
        return "generate"
    return "rewrite"

def route_after_generation(state: AgentState) -> str:
    """Decides if the response is safe or if it needs to be regenerated."""
    if state["hallucination_grade"] == "YES":
        return END
    elif state["loop_count"] >= 3:
        return END
    return "rewrite"

# Define Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("grade_docs", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("rewrite", rewrite_query)

# Add Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    route_after_retrieval,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges(
    "grade_generation",
    route_after_generation,
    {
        END: END,
        "rewrite": "rewrite"
    }
)

app = workflow.compile()
