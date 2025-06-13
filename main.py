from fastapi import FastAPI, Request
from research_analyst.workflow import app as langgraph_app

api = FastAPI()

@api.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    user_message = data.get("message")
    result = langgraph_app.invoke({"messages": [user_message]})
    return {"response": result}