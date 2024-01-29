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

## Demo

- **Query:** What's LangSmith?
  - **Classification:** Technical/General
  - **Agent:** Technical Agent
  - **Tool:** LangSmith Search Tool
  - **Response:** "LangSmith is a tool developed by LangChain..."

- **Query:** What's the weather in Lyon, France?
  - **Classification:** Technical/General
  - **Agent:** Technical Agent
  - **Tool:** Tavily Search Tool
  - **Response:** "The weather in Lyon, France is currently partly cloudy with a temperature of 52°F (11°C). The wind speed is 2.3 mph (3.7 km/h)"

- **Query:** What's 9+2?
  - **Classification:** Math
  - **Agent:** Math Agent
  - **Tool:** Basic Calculator Tool
  - **Response:** "The sum of 9 and 2 is 11."

- **Query:** What's the solution to 3x+5=14?
  - **Classification:** Math
  - **Agent:** Math Agent
  - **Tool:** Equation Solver Tool
  - **Response:** "I apologize, but the equation solver feature is currently under development. I am unable to provide you with the solution to the equation 3x+5=14 at the moment."
