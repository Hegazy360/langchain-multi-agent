# LangChain Multi-Agent Multi-Tool System POC

This Python application is a proof of concept (POC) for a basic multi-agent, multi-tool system using LangChain, OpenAI, Tavily, FAISS, and incorporating LangSmith for enhanced monitoring, logging, and debugging. It demonstrates specialized agents for handling technical and math-related queries with advanced input classification and efficient information retrieval.

## Features

- **Technical Agent:** Leverages LangChain, OpenAI, Tavily Search, and FAISS for handling technical queries.
- **Math Agent:** Specialized in math-related questions, equipped with a basic calculator and an equation solver tool.
- **LLM Classification:** Utilizes a language model (LLM) to classify user queries as 'math' or 'general/technical', directing them to the appropriate agent.
- **Embeddings & FAISS:** Employs OpenAI Embeddings and FAISS for efficient information retrieval.
- **Tavily Search:** Integrated as a search tool to enhance the agent's data access and processing capabilities.
- **LangSmith:** Used for monitoring, logging, and debugging to ensure smooth operation and maintenance of the system.

## Setup

### Dependencies

- LangChain
- OpenAI
- Tavily Search
- FAISS for information retrieval
- LangSmith for system monitoring

### How to Run

1. Use python venv `python -m venv .venv` & `source .venv/bin/activate`
2. Install the dependencies `pip install -r requirements.txt`
3. Set the necessary environment variables using a `.env` file.
4. Run the Python script using the command `python app.py`.
