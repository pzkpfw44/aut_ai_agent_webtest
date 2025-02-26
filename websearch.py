import openai
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI

# Initialize web search tool
search_tool = SerpAPIWrapper()

# Define a simple toolset
tools = [
    Tool(
        name="Web Search",
        func=search_tool.run,
        description="Useful for answering questions about current events and gathering information from the web."
    )
]

# Initialize AI agent with memory (optional)
llm = ChatOpenAI(model="gpt-4", temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent with a user-defined goal
goal = "Find the latest advancements in autonomous AI agents and summarize them."
response = agent.run(goal)

# Save the response to a file
with open("agentic_ai_report.txt", "w") as file:
    file.write(response)

print("Agent task complete. Results saved to 'agentic_ai_report.txt'.")
