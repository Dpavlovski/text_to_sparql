import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_ollama import ChatOllama

load_dotenv()
api_key = os.getenv("https://llama3.finki.ukim.mk/api/chat")
model = os.getenv("OLLAMA_MODEL")

search = TavilySearchResults()
search.invoke("what is the weather in SF")

llm = ChatOllama(model=model)
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

tools = [search]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "hi!"})

# def get_system_messages() -> ModelRequest:
#     return ModelRequest(
#         parts=[
#             SystemPromptPart(
#                 content='You are a helpful AI assistant that knows how to code and execute commands.',
#                 part_kind='system-prompt'
#             )
#         ]
#     )
#
#
# @agent.tool
# async def handle_alert(
#         ctx: RunContext[str],
#         alert: str
# ):
#     """Handles server alerts by performing linux commands.
#
#     :param ctx:
#     :param task:
#     :return:
#     """
#
#     response = await alert_handler(alert)
#     return response
