from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def create_first_agent():
    message_history = ChatMessageHistory()

    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")

    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    vector = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )

    search = TavilySearchResults()

    tools = [search, retriever_tool]

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history


@tool
def basic_calculator(query):
    """Basic calculator tool"""
    try:
        result = eval(query)
        return f"The result is {result}"
    except (SyntaxError, NameError) as e:
        return f"Sorry, I couldn't calculate that due to an error: {e}"


@tool
def equation_solver(query):
    """Equation solver tool"""
    # Basic equation solver (placeholder)
    # Implement specific logic for solving equations
    return "Equation solver: This feature is under development."


def create_second_agent():
    # Configuration for the second agent specialized in math questions
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Define tools for the second agent
    tools = [basic_calculator, equation_solver]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are specialized in solving math-related questions. Return the answer to the user's question."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


def get_agent(user_input):
    # Define the prompt template
    template = "You are a helpful assistant. Classify the user input as either 'math' if it's math-related or 'general/technical' otherwise. respond directly with the classification.\nQuestion: {question}\n"

    # Create a prompt
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Instantiate the OpenAI model
    llm = OpenAI(temperature=0)  # You can specify model_name as needed

    # Create an LLMChain with the prompt and the language model
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the chain with the user input
    response = llm_chain.invoke({"question": user_input})

    return response['text']


def main():
    load_dotenv()

    # Create agents
    technical_agent = create_first_agent()
    math_agent = create_second_agent()

    # Example user input
    user_input = "Hello! What's the solution to 3x+5=14?"

    # Invoke the agent decider
    response = get_agent(user_input)

    if response.strip().lower() == "math":
        print("invoking math agent")
        # Run the math agent
        math_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "test"}},
        )
    else:
        print("invoking technical agent")
        # Run the technical agent
        technical_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "test"}},
        )

# Run the main function
if __name__ == '__main__':
    main()
