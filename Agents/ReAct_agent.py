from typing import TypedDict, Annotated,Sequence
from langchain_core.messages import BaseMessage # the foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to llm after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage #Message for providing instructions to the llm
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from IPython.display import Image,display


load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

@tool  # type: ignore
def add(a: int,b: int):
    """This is an addition function that adds two numbers"""

    return a + b

tools = [add]

model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)

def model_call(state:AgentState)-> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistance, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"]) # type: ignore
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:  # type: ignore
        return "end"
    else:
        return "continue"
    


graph = StateGraph(AgentState)

graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our_agent")

app = graph.compile()

# Helper function
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","add 3 + 4 then add 2 to the result")]}
print_stream(app.stream(inputs,stream_mode="values"))

#____________________
# from PIL import Image as PILImage
# from io import BytesIO

# # Get the PNG bytes from your graph
# img_bytes = app.get_graph().draw_mermaid_png()

# # Convert bytes to image and show in default image viewer
# img = PILImage.open(BytesIO(img_bytes))
# img.show()  # This opens it in a default image viewer (popup)

#____________________