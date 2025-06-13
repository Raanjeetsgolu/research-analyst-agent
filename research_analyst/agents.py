import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import search_tool, python_repl_tool
from .prompts import make_system_prompt
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_model():
    return ChatOpenAI(model="gpt-4")

def create_research_agent():
    return create_react_agent(
        get_openai_model(),
        tools=[search_tool],
        prompt=make_system_prompt(
            "You can only do research. You are working with a chart generator colleague."
        ),
    )

def create_chart_agent():
    return create_react_agent(
        get_openai_model(),
        tools=[python_repl_tool],
        prompt=make_system_prompt(
            "You can only generate charts. You are working with a researcher colleague."
        ),
    )