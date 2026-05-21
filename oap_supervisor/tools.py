"""Tools for the agents."""

import json
import os
import requests
from langchain_core.tools import tool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load and process PDF
PDF_PATH = os.path.join("oap_supervisor","data", "Our Services.pdf")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=100,
)
doc_splits = text_splitter.split_documents(docs)

# Create vector store
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "search_company_policies",
    "Search and return information about company policies"
)


# Navigation tools
@tool
def optimize_route(start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> str:
    """Get optimized route between two coordinates. Returns route details including steps, distance, and duration."""
    api_key = os.getenv("OPENROUTE_API_KEY")
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [
            [start_lon, start_lat],
            [end_lon, end_lat]
        ],
    }
    
    response = requests.post(
        "https://api.openrouteservice.org/v2/directions/driving-car",
        headers=headers,
        json=body
    )
    
    data = response.json()
    return json.dumps(data, indent=2)


@tool
def get_navigation_step(route_data: str, step_number: int) -> str:
    """Extract specific navigation step from route data. Provide the route JSON and step number."""
    try:
        data = json.loads(route_data)
        segments = data['routes'][0]['segments']
        all_steps = []
        for segment in segments:
            all_steps.extend(segment['steps'])
        
        if 0 <= step_number < len(all_steps):
            step = all_steps[step_number]
            return f"Step {step_number + 1}/{len(all_steps)}: {step['instruction']} - Distance: {step['distance']}m, Duration: {step['duration']}s"
        else:
            return f"Invalid step number. Route has {len(all_steps)} steps total."
    except Exception as e:
        return f"Error parsing route data: {str(e)}"


@tool
def get_route_summary(route_data: str) -> str:
    """Get summary of the entire route including total distance and duration."""
    try:
        data = json.loads(route_data)
        route = data['routes'][0]
        summary = route['summary']
        return f"Total Distance: {summary['distance']/1000:.2f}km, Duration: {summary['duration']/60:.2f} minutes"
    except Exception as e:
        return f"Error parsing route data: {str(e)}"