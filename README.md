Multi-agent orchestration system powered by **LangGraph** Supervisor Agent to coordinate multiple agents, managed with **UV**. 
It uses specialized agents for customer support, logistics optimization, forecasting, search and analytics.
* * *

Quick Start
--------------

### 1\. **Clone the Repository**

```bash
git clone <This-repo-url>
cd <cloned-repo-folder>
```

### 2\. **Set Up Environment Variables**

Create a `.env` file in the project root and add the following keys:

```bash

LANGCHAIN_PROJECT="default"
LANGCHAIN_API_KEY=""
LANGCHAIN_TRACING_V2=true

GOOGLE_API_KEY=""
OPEN_ROUTE_API_KEY=""
TAVILY_API_KEY=""
SUPABASE_URL=""
# Ensure this is your Supabase Service Role key
SUPABASE_KEY=""
```

> ⚠️ Make sure you have valid API keys for each service.

* * *

### 3\. **Install Dependencies**

This project uses **UV** for dependency management.  
If you don’t have UV installed:

```bash
pip install uv
```

Then, install all required dependencies:

```bash
uv sync
```

* * *

### 4\. **Run the Agent**

Start the agent using the following command:

```bash
uv run langgraph dev --allow-blocking
```

> ⏳ The first time you run this, the embeddings model will be downloaded automatically.  
> Wait until the embeddings are fully created before interacting with the agent.

* * *

### 5\. **Interact via LangGraph Studio**

Once the agent is running, open the **LangGraph Studio UI** to interact with it:

🔗 **Studio URL:**  
[https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)

* * *

Notes
--------

*   Ensure your environment variables are properly configured before running the agent.
*   The agent automatically handles model downloading, embedding creation, and setup on the first run.
*   For troubleshooting, check the terminal logs while the agent starts.
