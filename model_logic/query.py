import os
from collections import defaultdict
from dotenv import load_dotenv

import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chromaPDF"
embedding_func = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)
llm = ChatOpenAI()
memory = InMemorySaver()

# function used to call model and save past/current MessagesState
def call_model(state: MessagesState):
    query = state["messages"][-1].text()
    results = db.similarity_search_with_score(query, k=5)

    sources = format_sources(results)
    context = build_context(results)

    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt_template.format(
        context=context,
        question=query,
        sources=sources
    )
    # add the context to the invocation 
    response = llm.invoke(state["messages"] + [HumanMessage(prompt)])

    # return the overall state plus the new response
    return {"messages": state["messages"] + [response]}

# Build the LangGraph graph
graph = StateGraph(MessagesState)

# add chat node to be able to call the model
graph.add_node("chat", call_model)

# define the entry point for the graph and compile it
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
compiled_graph = graph.compile(checkpointer=memory)


TEMPLATE = """
You are a helpful and knowledgeable assistant. Your job is to answer user questions using only the information provided in the context below and the conversation history.
You can also answer general questions if asked.

Instructions:
- Be specific and detailed in your answers.
- If the question is vague, ask for a more specific clarification.
- If the context does not provide any relevant information, tell the user to ask a more specific question.
- Only answer if you are absolutely sure you are correct. Otherwise, specify that you are unsure and provide reasoning.
- At the end of your response include a section for the sources you used in this format: 
    Sources: 
        - <source>, Page(s): <page_numbers>

Context:
{context}

Sources:
{sources}

---

Now, answer the following question using only the context. Do not mention the context in your response unless explicitly asked.
Make sure to cite your sources. Don't be afraid to ask for clarification if needed.

Question:
{question}
"""

def format_sources(results):
    """
    Formats the sources and page numbers from the search results.
    """
    source_pages = defaultdict(set)
    for doc, _ in results:
        source = doc.metadata.get("source", None)
        page = doc.metadata.get("page", 0) + 1
        source_pages[source].add(page)

    sources = []
    for key, value in source_pages.items():
        sources.append(f'Source: {key}, Pages: {sorted(value)} | ')

    return "".join(sources)

def build_context(results):
    """
    Builds the context string from the retrieved documents.
    """
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])


def langGraph_chat(user_query: str, thread_id: str = "1"):
    """
    Handles a chat turn using LangGraph and InMemorySaver for memory.
    Each thread_id gets its own conversation history.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    message = HumanMessage(content=user_query)

    # execute the graph to save the state
    compiled_graph.invoke(config=config, input={"messages": [message]})

    return compiled_graph.get_state(config).values['messages'][-1].text()

if __name__ == "__main__":
    # Example usage
    user_query = "What is it about?"
    thread_id = "user123"  # Unique identifier for the conversation thread
    print(langGraph_chat(user_query, thread_id))
