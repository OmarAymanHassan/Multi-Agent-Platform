import json
import requests
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import Literal
from pydantic import BaseModel, Field
import os


import json
import requests
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os


from langchain_core.messages import convert_to_messages

from langgraph.prebuilt import create_react_agent


#load API keys from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_ROUTE_KEY = os.getenv("OPEN_ROUTE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print(GOOGLE_API_KEY)
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")



        # Initialize Gemini models
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Initialize Gemini models
grader_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Load PDF
# Configuration
PDF_PATH = os.path.join("oap_supervisor", "Our Services.pdf") # Path to your company policies PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()



# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,  # Increased for better context
    chunk_overlap=100,
)

doc_splits = text_splitter.split_documents(docs)

print(f"Created {len(doc_splits)} document chunks")

# 2. Create a retriever tool
print("Creating vector store and retriever...")

# Create vector store
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)


# Test the vector store
test_result = vectorstore.similarity_search("order cancellation", k=3)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 relevant chunks
)

# Create retriever tool

retriever_tool = create_retriever_tool(
    retriever,
    "search_company_policies",
    "Search and return information about company policies"
)


def generate_query_or_respond(state: MessagesState):
    """Process the user's question and provide an answer or search for information."""
    
    # Get the actual user question from the messages
    user_question = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'type') and msg.type == 'human':
            user_question = msg.content
            break
        elif isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    if not user_question:
        return {"messages": [{"role": "assistant", "content": "I didn't receive a clear question. How can I help you?"}]}
    
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful customer service agent for a talabat company. "
            "Answer the user's question about account creation, orders, policies, and services. "
            "Use the search_company_policies tool when you need specific policy information. "
            "Provide complete, helpful answers."
        )
    }
    
    messages = [system_message, {"role": "user", "content": user_question}]
    response = llm.bind_tools([retriever_tool]).invoke(messages)
    return {"messages": [response]}


# 4. Grade documents
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a customer question. \n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the customer question: {question} \n"
    "If the document contains information that helps answer the customer's question about "
    "orders, policies, delivery, and services and accounts.  \n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )



grader_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    
    # Get the last tool message content
    tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
    if not tool_messages:
        return "rewrite_question"
    
    context = tool_messages[-1].content
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    # Use structured output with Gemini
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    
    # Simple keyword-based grading for Gemini (as it doesn't support structured output like GPT)
    response_text = response.content.lower()
    if "yes" in response_text and "no" not in response_text:
        return "generate_answer"
    else:
        return "rewrite_question"
    

# 5. Rewrite question
REWRITE_PROMPT = (
    "You are helping a customer service agent improve search queries. "
    "The customer asked a question, but the search didn't return relevant results. "
    "Rewrite the question to be more specific and likely to find relevant policy information.\n"
    "Original question: {question}\n"
    "Provide an improved search query that focuses on key terms related to order and accounts:"
)






def rewrite_question(state: MessagesState):
    """Rewrite the original customer question for better search results."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    # Return a new user message with the rewritten question
    # return {"messages": [{"role": "user", "content": response.content}]}
    return {"messages": [response]}



# 6. Generate answer
GENERATE_PROMPT = (
    "You are a customer service agent for a talabat company. "
    "Use the following information from our company policies to answer the customer's question. "
    "Be accurate, professional, and helpful. If the information doesn't fully answer the question, "
    "acknowledge what you can answer and offer to help further.\n\n"
    "Customer Question: {question}\n\n"
    "Relevant Policy Information: {context}\n\n"
    "Provide a clear and helpful response:"
)


def generate_answer(state: MessagesState):
    """Generate a final answer based on retrieved context."""
    question = state["messages"][0].content
    
    # Get the last tool message content
    tool_messages = [msg for msg in state["messages"] if msg.type == "tool"]
    context = tool_messages[-1].content if tool_messages else "No relevant information found."
    
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [AIMessage(content=response.content)]}



workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge(START, "generate_query_or_respond")

# Conditional edge: decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    }
)

# Conditional edge: grade documents
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question"
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile the graph
customer_service_graph = workflow.compile(name = "customer_service_agent")

print("Agent graph compiled successfully!")





google_api_key = GOOGLE_API_KEY
api_key = OPEN_ROUTE_KEY


@tool
def optimize_route(start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> str:
    """Get optimized route between two coordinates. Returns route details including steps, distance, and duration."""
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

@tool 
def get_weather() -> str:
    """Get current weather and traffic conditions for the route."""
    # make a list of possiple reposne and choose randomly
    # This is a placeholder implementation. Replace with actual weather API call.
    reponses = ["it is suuny", "it is cloudy", "it is raining", "it is snowing"]
    # For simplicity, returning a static response
    # return random from responses 
    return "The current weather is sunny "

@ tool 
def get_traffic() -> str:
    """Get current traffic conditions for the route."""
    # This is a placeholder implementation. Replace with actual traffic API call.
    responses = ["Light traffic", "Moderate traffic", "Heavy traffic"]
    # For simplicity, returning a static response
    return "The current traffic condition is light."



# Define tools
tools = [optimize_route, get_navigation_step, get_route_summary, get_weather, get_traffic]




# Initialize model with tools
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.1,
    google_api_key=google_api_key
).bind_tools(tools)

# Define state with chat history
class State(TypedDict):
    messages: Annotated[list, add_messages]
    chat_history: List[dict]  # Explicit chat history storage
    route_data: str
    current_step: int


def format_chat_history(chat_history: List[dict]) -> str:
    """Format chat history for inclusion in prompt"""
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for entry in chat_history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)



# Model node with chat history
def model_node(state: State) -> State:
    # Format chat history
    chat_history_str = format_chat_history(state.get("chat_history", []))
    
    # Create system message with chat history
    system_prompt = f"""
    You are a navigation assistant that helps users with route planning and navigation and weather and traffic.
    if the user askes about waether or traffic , repsonse with any logical answer
    Tools:
      optimize_route: Get an optimized route between two coordinates (longitude, latitude pairs)
      get_navigation_step: Extract a specific navigation step from route data
      get_route_summary: Get a summary of the route including total distance and duration
      get_weather: provide weather information for a specific location , no need to coordianates
      get_traffic: provide traffic information for a specific locations , no need to coordianates   

    Current Route Data: {state.get('route_data', 'No route loaded yet')}
    Current Step: {state.get('current_step', 0)}

    Conversation History:
    {chat_history_str}

    Instructions:
      - When asked about navigation, first use optimize_route to get the route data
      - when asked about weather or traffic, just answer with any logical answer 
      - Store the route data and use it to answer subsequent questions
      - write the full route plan
      - Provide clear, formatted responses 
      - Be conversational and helpful
      - Remember the conversation context from the chat history
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]
    
    res = model.invoke(messages)
    
    # Update chat history
    new_chat_history = state.get("chat_history", []).copy()
    
    # Add the latest human message to chat history
    if state["messages"]:
        last_human_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                last_human_msg = msg
                break
        
        if last_human_msg:
            new_chat_history.append({
                "role": "Human",
                "content": last_human_msg.content
            })
    
    # Add AI response to chat history
    new_chat_history.append({
        "role": "Assistant",
        "content": res.content
    })
    
    # Keep only last N exchanges (configurable)
    max_history_length = 10  # Keep last 10 exchanges
    if len(new_chat_history) > max_history_length * 2:
        new_chat_history = new_chat_history[-(max_history_length * 2):]
    
    return {
        "messages": [AIMessage(content=res.content)],
        "chat_history": new_chat_history
    }


# Build the graph
builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

# Compile with memory
navigation_graph = builder.compile( name = "navigation_agent")
#checkpointer=MemorySaver(),

navigation_graph = create_react_agent(
    model = model,
    tools = tools,
    name = "navigation_agent",
    prompt = "You are a navigation assistant that helps users with route planning and navigation and weather and traffic. If the user asks about weather or traffic, respond with any logical answer.",
)


# Initialize Gemini models
llm_google = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

from typing import Annotated, TypedDict , Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

class State(TypedDict):
  messages: Annotated[list, add_messages]
  analytics: Optional[str]
  csv_file:Optional[str]

# Initialize Gemini models
llm_google_2 = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def analytics_agent(state:State)->State:
  msgs = state['messages'][0].content
  df_path = state.get('csv_file')
  df_path = "logistics_inventory_dummy_data.csv"
  
  # in case of absence of csv file 

  # if state.get('csv_file') is None:
  if df_path is None or df_path == "":
    print("Now we are going to analyze the data that provided by the user")
    
    system_prompt = ChatPromptTemplate.from_template("You are an expert Data Analyst. Analyze the data  given by the user and provide insights in the best way  and just return the answer . question : {question}")
    #prompt = system_prompt.invoke({"question" : msgs})
    chain = system_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": msgs})
    return {"messages":state["messages"],"analytics": result, "csv_file": df_path} 

  else: 

    # in case of presence of csv file 
    
    # df = pd.read_csv(state['csv_file'])
    df = pd.read_csv(df_path)
    print("--- Analyzing the csv file ---")


    system_prompt = "You are an expert Data Analyst. Analyze the data and provide insights in the best way"

    df_agent = create_pandas_dataframe_agent(
      llm=llm_google_2,
      df=df,
      system_prompt=system_prompt,
      verbose=True,
      allow_dangerous_code=True
    )
    result = df_agent.run(msgs)

    return {"messages":state["messages"] ,"analytics": result, "csv_file": df_path} 


builder = StateGraph(State)


builder.add_node("analytics_tool", analytics_agent)
builder.add_edge(START, 'analytics_tool')
builder.add_edge('analytics_tool', END)
analytical_agent_graph = builder.compile(name="analytical_agent")




from typing import List, Dict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy

# Define the RandomForestARModel class (unchanged)
class RandomForestARModel:
    """
    Autoregressive forecasting with Random Forests
    """
    def __init__(self, n_lags=1, max_depth=3, n_estimators=20, random_state=123,
                 log_transform=False, first_differences=False, seasonal_differences=None):
        """
        Args:
            n_lags: Number of lagged features to consider in autoregressive model
            max_depth: Max depth for the forest's regression trees
            random_state: Random seed to pass to random forest
            log_transform: Whether the input should be log-transformed
            first_differences: Whether the input should be singly differenced
            seasonal_differences: Seasonality to consider, if 'None' then no seasonality is presumed
        """
        self.n_lags = n_lags
        self.model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
        self.log_transform = log_transform
        self.first_differences = first_differences
        self.seasonal_differences = seasonal_differences

    def fit(self, y):
        """
        Args:
            y: training data (numpy array or pandas series/dataframe)
        """
        y_df = pd.DataFrame(y)
        self.y_df = deepcopy(y_df)
        
        if self.log_transform:
            y_df = np.log(y_df)
            self.y_logged = deepcopy(y_df)
        
        if self.first_differences:
            y_df = y_df.diff().dropna()
            self.y_diffed = deepcopy(y_df)
        
        if self.seasonal_differences is not None:
            y_df = y_df.diff(self.seasonal_differences).dropna()
            self.y_diffed_seasonal = deepcopy(y_df)
        
        Xtrain = pd.concat([y_df.shift(t) for t in range(1, self.n_lags+1)], axis=1).dropna()
        self.Xtrain = Xtrain
        ytrain = y_df.loc[Xtrain.index, :]
        self.ytrain = ytrain
        self.model.fit(Xtrain.values, ytrain.values.reshape(-1))

    def sample_forecast(self, n_periods=1, n_samples=10000, random_seed=123):
        """
        Draw forecasting samples by randomly drawing from all trees in the forest per forecast period
        Args:
            n_periods: Amount of periods to forecast
            n_samples: Number of samples to draw
            random_seed: Random seed for numpy
        """
        samples = self._perform_forecast(n_periods, n_samples, random_seed)
        output = self._retransform_forecast(samples, n_periods)
        return output

    def _perform_forecast(self, n_periods, n_samples, random_seed):
        samples = []
        np.random.seed(random_seed)
        for i in range(n_samples):
            Xf = np.concatenate([self.Xtrain.iloc[-1, 1:].values.reshape(1, -1),
                                 self.ytrain.iloc[-1].values.reshape(1, 1)], 1)
            forecasts = []
            for t in range(n_periods):
                tree = self.model.estimators_[np.random.randint(len(self.model.estimators_))]
                pred = tree.predict(Xf)[0]
                forecasts.append(pred)
                Xf = np.concatenate([Xf[:, 1:], np.array([[pred]])], 1)
            samples.append(forecasts)
        return samples

    def _retransform_forecast(self, samples, n_periods):
        full_sample_tree = []
        for samp in samples:
            draw = np.array(samp)
            if self.seasonal_differences is not None:
                result = list(self.y_diffed.iloc[-self.seasonal_differences:].values)
                for t in range(n_periods):
                    result.append(result[t] + draw[t])
                result = result[self.seasonal_differences:]
            else:
                result = []
                for t in range(n_periods):
                    result.append(draw[t])
            
            y_for_add = self.y_logged.values[-1] if self.log_transform else self.y_df.values[-1]
            if self.first_differences:
                result = y_for_add + np.cumsum(result)
            if self.log_transform:
                result = np.exp(result)
            full_sample_tree.append(result.reshape(-1, 1))
        return np.concatenate(full_sample_tree, 1)




# Define MessagesState for consistency with other agents
from langgraph.graph import StateGraph, START, END, MessagesState



# Define forecasting function
def train_RF(state: MessagesState) -> MessagesState:
    """
    Perform forecasting using RandomForestARModel and return a formatted response.
    """
    # Default parameters
    default_params = {
        "CSV_file": "Alcohol_Sales.csv",
        "n_periods": 48,
        "n_samples": 10000,
        "test_size": 48,
        "n_lags": 2,
        "log_transform": True,
        "first_differences": True,
        "seasonal_differences": 12
    }

    # Extract parameters from the last human message
    try:
        last_message = [msg for msg in state["messages"] if msg.type == "human"][-1]
        query = last_message.content.lower()
        params = default_params.copy()
        
        # Parse query for custom parameters (simple keyword-based parsing)
        if "periods" in query:
            try:
                n_periods = int(query.split("periods")[1].split()[0])
                params["n_periods"] = n_periods
            except:
                pass
        if "samples" in query:
            try:
                n_samples = int(query.split("samples")[1].split()[0])
                params["n_samples"] = n_samples
            except:
                pass
        if "file" in query:
            try:
                csv_file = query.split("file")[1].split()[0]
                params["CSV_file"] = csv_file
            except:
                pass
    except:
        params = default_params

    try:
        # Load and preprocess data
        df = pd.read_csv(params["CSV_file"])
        df.columns = ["date", "sales"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        if params["n_periods"] == -1:
            df_train = df.iloc[:-params["test_size"]]
            df_test = df.iloc[-params["test_size"]:]
            params["n_periods"] = len(df_test)
        else:
            df_train = df
            df_test = None

        # Train model
        model = RandomForestARModel(
            n_lags=params["n_lags"],
            log_transform=params["log_transform"],
            first_differences=params["first_differences"],
            seasonal_differences=params["seasonal_differences"]
        )
        model.fit(df_train["sales"])

        # Generate forecast
        predictions_forest = model.sample_forecast(
            n_periods=params["n_periods"],
            n_samples=params["n_samples"]
        )

        # Calculate statistics
        means_forest = np.mean(predictions_forest, 1)
        lowers_forest = np.quantile(predictions_forest, 0.05, 1)
        uppers_forest = np.quantile(predictions_forest, 0.95, 1)

        # Format response
        forecast_dates = pd.date_range(start=df.index[-1], periods=params["n_periods"] + 1, freq="MS")[1:]
        forecast_summary = "\n".join([
            f"{date.strftime('%Y-%m')}: Mean: {mean:.2f}, 90% CI: ({lower:.2f}, {upper:.2f})"
            for date, mean, lower, upper in zip(forecast_dates, means_forest, lowers_forest, uppers_forest)
        ])
        
        response = (
            f"Forecast for {params['n_periods']} periods:\n"
            f"{forecast_summary}\n"
            f"Model parameters: n_lags={params['n_lags']}, log_transform={params['log_transform']}, "
            f"first_differences={params['first_differences']}, seasonal_differences={params['seasonal_differences']}"
        )
        
        return {
            "messages": [AIMessage(content=f"Answer: {response}\nAdditional Info: Would you like to adjust the forecast parameters or visualize the results?")]
        }

    except FileNotFoundError:
        return {
            "messages": [AIMessage(content=f"Answer: Error: Could not find the file '{params['CSV_file']}'.\nAdditional Info: Please provide a valid CSV file path.")]
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Answer: Error during forecasting: {str(e)}\nAdditional Info: Please check the input parameters and try again.")]
        }


# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("forecast", train_RF)
workflow.add_edge(START, "forecast")
workflow.add_edge("forecast", END)

# Compile the graph
forecast_agent_graph = workflow.compile(name="forecasting_agent")





from langchain_tavily import TavilySearch



web_search = TavilySearch(max_results=1, tavily_api_key =TAVILY_API_KEY )




research_agent = create_react_agent(
    model=llm_google,
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)









from langgraph_supervisor import create_supervisor


def graph():
    supervisor = create_supervisor(
        model=llm_google,
        agents=[customer_service_graph, navigation_graph,analytical_agent_graph,forecast_agent_graph, research_agent],
        prompt=(
            "You are a supervisor managing three agents:\n"
            "- a customer_service_agent. Assign customer service tasks to this agent\n"
            "- a navigation_agent. Assign route and distance and weather and traffic related tasks to this agent\n"
            "- a analytical_agent. Assign anaytics and stastics related tasks to this agent\n"
            "- a forecast_agent. Assign forecasting tasks to this agent\n"
            "- a research_agent. Assign research tasks to this agent\n\n"
            "INSTRUCTIONS:\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "generate the final answer based on the agent's response.\n"
            "if the customer request is not related to any of the agents, respond with any logical answer\n"
            "if any agent failed to return the answer, respond with any logical answer\n"

        ),
        add_handoff_back_messages=False,
        output_mode="full_history",
    )
    return supervisor