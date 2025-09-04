# graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain.schema import SystemMessage, HumanMessage
from agents import llm, search_tool


# ---------------- Shared State ----------------
class MeetingState(TypedDict, total=False):
    # Inputs
    company: str
    objective: str
    attendees: str
    duration: int
    focus: str

    # Outputs
    context: str
    industry: str
    strategy: str
    brief: str


# ---------------- Nodes ----------------
def context_node(state: MeetingState) -> MeetingState:
    search_results = search_tool.run(f"Latest news and background about {state['company']}")
    messages = [
        SystemMessage(content="You are a Meeting Context Specialist. Summarize company details and key background."),
        HumanMessage(content=f"""
        Company: {state['company']}
        Objective: {state['objective']}
        Attendees: {state['attendees']}
        Duration: {state['duration']} minutes
        Focus areas: {state['focus']}

        Search results about the company:
        {search_results}

        Please combine these results with your own knowledge to create a meeting context summary.
        """)
    ]
    result = llm.invoke(messages)
    state["context"] = result.content
    return state


def industry_node(state: MeetingState) -> MeetingState:
    search_results = search_tool.run(f"Latest industry trends and competitors for {state['company']}")
    messages = [
        SystemMessage(content="You are an Industry Analyst. Provide industry analysis, competitors, opportunities, and threats."),
        HumanMessage(content=f"""
        Context about {state['company']}:
        {state['context']}

        Search results on industry:
        {search_results}

        Please combine the context and search results to produce a thorough industry analysis.
        """)
    ]
    result = llm.invoke(messages)
    state["industry"] = result.content
    return state


def strategy_node(state: MeetingState) -> MeetingState:
    search_results = search_tool.run(f"Meeting strategies and talking points for {state['company']} in its industry")
    messages = [
        SystemMessage(content="You are a Meeting Strategist. Create agenda, talking points, and strategies."),
        HumanMessage(content=f"""
        Company: {state['company']}
        Objective: {state['objective']}
        Duration: {state['duration']} minutes
        Focus areas: {state['focus']}

        Company context:
        {state['context']}

        Industry insights:
        {state['industry']}

        Extra search results for inspiration:
        {search_results}

        Please create a time-boxed agenda, key talking points, discussion questions,
        and strategies to address the focus areas.
        """)
    ]
    result = llm.invoke(messages)
    state["strategy"] = result.content
    return state


def brief_node(state: MeetingState) -> MeetingState:
    search_results = search_tool.run(f"Executive summary style insights for meeting with {state['company']}")
    messages = [
        SystemMessage(content="You are a Communication Specialist. Create a clear, concise executive brief."),
        HumanMessage(content=f"""
        Objective: {state['objective']}
        Attendees: {state['attendees']}

        Company context:
        {state['context']}

        Industry insights:
        {state['industry']}

        Meeting strategy:
        {state['strategy']}

        Recent relevant info:
        {search_results}

        Please produce a one-page executive summary with:
        - Meeting objective
        - Attendees & roles
        - Company background
        - Industry insights
        - Top 3-5 strategic goals
        - Key talking points with supporting data
        - Anticipated questions with answers
        - Actionable recommendations
        """)
    ]
    result = llm.invoke(messages)
    state["brief"] = result.content
    return state


# ---------------- Controller + Routing ----------------
def controller(state: MeetingState) -> MeetingState:
    """Pass state forward, routing handled separately."""
    return state


def route(state: MeetingState) -> str:
    """Decide which node to call next based on what’s missing in state."""
    if "context" not in state:
        return "context"
    elif "industry" not in state:
        return "industry"
    elif "strategy" not in state:
        return "strategy"
    elif "brief" not in state:
        return "brief"
    else:
        return END


# ---------------- Graph Workflow ----------------
graph = StateGraph(MeetingState)

# Add worker nodes
graph.add_node("context", context_node)
graph.add_node("industry", industry_node)
graph.add_node("strategy", strategy_node)
graph.add_node("brief", brief_node)

# Add controller node
graph.add_node("controller", controller)

# Conditional edges from controller
graph.add_conditional_edges(
    "controller",
    route,   # routing function
    {
        "context": "context",
        "industry": "industry",
        "strategy": "strategy",
        "brief": "brief",
        END: END,
    },
)

# After each worker node, return to controller
graph.add_edge("context", "controller")
graph.add_edge("industry", "controller")
graph.add_edge("strategy", "controller")
graph.add_edge("brief", "controller")

# Start at controller
graph.set_entry_point("controller")

# Compile final workflow
meeting_workflow = graph.compile()
