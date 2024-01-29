import chainlit as cl
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain.globals import set_debug
import os


@cl.on_chat_start
def start() -> None:
    llm = ChatVertexAI(
        model_name="gemini-pro",
        streaming=True,
        convert_system_message_to_human=True,
    )
    prompt = hub.pull("hwchase17/structured-chat-agent")
    tools = []
    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[cl.AsyncLangchainCallbackHandler()],
    )
    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def main(message: cl.Message) -> None:
    if os.environ.get("DEBUG"):
        set_debug(True)
    agent: AgentExecutor = cl.user_session.get("agent_executor")
    res = await agent.ainvoke({"input": message.content})
    await cl.Message(content=res).send()
