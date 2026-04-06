import ast
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class ReActState(TypedDict):
    input: str
    agent_scratchpad: str
    final_answer: str
    steps: List[Dict[str, Any]]
    pending_action: Optional[Dict[str, Any]]
    route: Literal["is_action", "is_final"]
    tool_result: str
    iterations: int


REACT_SYSTEM = """You are a ReAct agent.
You must follow this loop exactly:
Thought -> Action -> Action Input -> Observation -> Thought -> ...

Rules:
1) Use only the tools listed below for factual or numerical tasks.
2) For multi-part questions, perform multiple tool calls as needed.
3) If a calculation is needed, call a math tool.
4) Stop only when all parts are answered.

Output format:
- If another tool is needed:
Thought: <brief reasoning>
Action: <exact tool name>
Action Input: <valid JSON object>

- If complete:
Thought: <brief reasoning>
Final Answer: <final response>
"""


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _build_llm():
    _load_env()
    if os.getenv("GROQ_API_KEY"):
        ChatGroq = getattr(importlib.import_module(
            "langchain_groq"), "ChatGroq")
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        ChatGoogleGenerativeAI = getattr(importlib.import_module(
            "langchain_google_genai"), "ChatGoogleGenerativeAI")
        return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"), temperature=0)
    ChatOllama = getattr(importlib.import_module(
        "langchain_ollama"), "ChatOllama")
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)


def _build_mcp_client():
    MultiServerMCPClient = getattr(importlib.import_module(
        "langchain_mcp_adapters.client"), "MultiServerMCPClient")

    root = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(root, "Tools")
    return MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": [os.path.join(tools_dir, "math_server.py")],
                "transport": "stdio",
            },
            "search": {
                "command": sys.executable,
                "args": [os.path.join(tools_dir, "search_server.py")],
                "transport": "stdio",
            },
            "weather": {
                "url": os.getenv("WEATHER_MCP_URL", "http://localhost:8000/mcp"),
                "transport": "streamable_http",
            },
        }
    )


def _tools_text(tools: List[Any]) -> str:
    lines: List[str] = []
    for t in tools:
        schema = t.args if hasattr(t, "args") else {}
        try:
            schema_str = json.dumps(schema, ensure_ascii=True)
        except Exception:
            schema_str = "{}"
        lines.append(
            f"- {t.name}: {getattr(t, 'description', '').strip()} | args_schema={schema_str}")
    return "\n".join(lines)


def _parse_action_payload(text: str) -> Optional[Dict[str, Any]]:
    action = None
    action_input = None
    final_answer = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            action = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("action input:"):
            action_input = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("final answer:"):
            final_answer = stripped.split(":", 1)[1].strip()

    if final_answer:
        return {"type": "final", "answer": final_answer}

    if not action:
        return None

    parsed_args: Dict[str, Any]
    if not action_input:
        parsed_args = {}
    else:
        try:
            parsed_args = json.loads(action_input)
            if not isinstance(parsed_args, dict):
                parsed_args = {"input": parsed_args}
        except Exception:
            try:
                parsed_literal = ast.literal_eval(action_input)
                if isinstance(parsed_literal, dict):
                    parsed_args = parsed_literal
                else:
                    parsed_args = {"input": str(parsed_literal)}
            except Exception:
                parsed_args = {"query": action_input}

    return {"type": "action", "tool": action, "args": parsed_args}


def _router(state: ReActState) -> Literal["tool_node", "__end__"]:
    if state["route"] == "is_action":
        return "tool_node"
    return END


async def build_graph():
    llm = _build_llm()
    client = _build_mcp_client()

    tool_list: List[Any] = []
    for server_name in ["search", "math", "weather"]:
        server_tools = await client.get_tools(server_name=server_name)
        tool_list.extend(server_tools)
    tools_map = {t.name: t for t in tool_list}
    tool_catalog = _tools_text(tool_list)

    async def react_node(state: ReActState) -> ReActState:
        prompt = (
            f"{REACT_SYSTEM}\n\n"
            f"Available tools:\n{tool_catalog}\n\n"
            f"User Input:\n{state['input']}\n\n"
            f"Scratchpad:\n{state['agent_scratchpad']}\n"
        )

        response = await llm.ainvoke([
            SystemMessage(content="Return only the required ReAct format."),
            HumanMessage(content=prompt),
        ])
        text = (response.content or "").strip()
        parsed = _parse_action_payload(text)

        state["iterations"] = state["iterations"] + 1

        if state["iterations"] > 25:
            state["final_answer"] = "Max iterations reached before final answer."
            state["route"] = "is_final"
            return state

        if not parsed:
            state["agent_scratchpad"] += f"\n{text}\n"
            state["final_answer"] = text if text else "No valid response produced by model."
            state["route"] = "is_final"
            return state

        if parsed["type"] == "final":
            state["agent_scratchpad"] += f"\n{text}\n"
            state["final_answer"] = parsed["answer"]
            state["pending_action"] = None
            state["route"] = "is_final"
            return state

        tool_name = parsed["tool"]
        tool_args = parsed["args"]
        state["pending_action"] = {"tool": tool_name, "args": tool_args}
        state["route"] = "is_action"
        state["steps"].append(
            {"thought_action": text, "action": tool_name, "action_input": tool_args})
        state["agent_scratchpad"] += f"\n{text}\n"
        return state

    async def tool_node(state: ReActState) -> ReActState:
        pending = state.get("pending_action") or {}
        tool_name = pending.get("tool")
        tool_args = pending.get("args", {})

        if tool_name not in tools_map:
            observation = f"Tool '{tool_name}' not found."
        else:
            try:
                observation = str(await tools_map[tool_name].ainvoke(tool_args))
            except Exception as exc:
                observation = f"Tool execution error: {exc}"

        state["tool_result"] = observation
        state["steps"].append({"observation": observation})
        state["agent_scratchpad"] += f"Observation: {observation}\n"
        state["pending_action"] = None
        return state

    graph_builder = StateGraph(ReActState)
    graph_builder.add_node("react_node", react_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "react_node")
    graph_builder.add_conditional_edges("react_node", _router)
    graph_builder.add_edge("tool_node", "react_node")

    return graph_builder.compile()


async def run_agent(query: str) -> ReActState:
    app = await build_graph()
    initial_state: ReActState = {
        "input": query,
        "agent_scratchpad": "",
        "final_answer": "",
        "steps": [],
        "pending_action": None,
        "route": "is_action",
        "tool_result": "",
        "iterations": 0,
    }
    return await app.ainvoke(initial_state)
