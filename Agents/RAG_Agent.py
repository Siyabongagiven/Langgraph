from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4o",temperature=0 # minimize hallucination
)

embeddings = OpenAIEmbeddings(
    # model="text-embedding-3-small"
    model="text-embedding-ada-002"
)

pdf_path = "Stock_Market_Performance_2024.pdf"

# Safety measure I have put for debugging purposes 
if not os.path.exists(pdf_path):
    raise FileExistsError(f"PDF file not found : {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# Check if the pdf is there

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"ERROR loading PDF : {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pages_split  = text_splitter.split_documents(pages) # applying the chunk process to our pages


persist_directory = r"C:\Users\Siyabonga\OneDrive\Desktop\Projects\Langgraph\Agents\storage"
collection_name = "stock_maket"

#if our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    print("Creating Chroma Vectorstore...")
    #Here, we actually create the chroma database using our embeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"ERROR setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever 

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5} #K is the amount of chunks to return
)

@tool
def retriever_tool(query:str)->str:
    """
    This tool searches and returns the information from the stock market performance 2024 documet.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the stock Market Performance 2024 document."
    
    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}:\n {doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state:AgentState)->AgentState:
    """Check id the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls) > 0  # type: ignore


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name:our_tool for our_tool in tools} # Create a dictionry of our tools

#LLM Agent

def call_llm(state:AgentState)->AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages':[message]}


# Retriever Agent
def take_action(state:AgentState)->AgentState:
    """Excecutes tool calls from the LLM's response."""
    
    tool_calls = state["messages"][-1].tool_calls # type: ignore
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t["name"]} with query : {t["args"].get('query','No query provided')}")

        if not t['name'] in tools_dict: #Check if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result= "Incorect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result lenth : {len(str(result))}")

        results.append(ToolMessage(tool_call_id = t['id'],name=t['name'],content=str(result)))

    print('Tools Execution Complete. Back to Model!')
    return {'messages':results}

graph = StateGraph(AgentState)

graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

graph.add_conditional_edges(
    'llm',
    should_continue, # type: ignore
    {
        True:'retriever_agent',
        False:END
    }
)

graph.add_edge('retriever_agent','llm')
graph.set_entry_point('llm')

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        print("Ready for input...")
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()