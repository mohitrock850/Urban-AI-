# src/agents.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

# --- LLM Setup ---
# Initialize the language model that will power all the agents
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# --- Planner Pydantic Model ---
class SimpleDesign(BaseModel):
    """A model to hold the DALL-E prompt for the site plan."""
    dalle_prompt: str = Field(description="A detailed DALL-E 3 prompt for generating a top-down, architectural site plan diagram.")

# --- Agent Definitions ---

def create_planner_agent():
    """
    Creates the Planner agent.
    This agent's job is to take the user's request, compliance rules, and any
    feedback from a previous failed attempt, and generate a new, improved DALL-E prompt.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert urban planner. Your task is to convert a user's request, compliance rules, and critique feedback into a single, detailed DALL-E prompt.
                The prompt must describe a top-down, architectural site plan diagram. It should describe clear, distinct shapes for buildings, green spaces, and infrastructure.
                If you receive feedback about a compliance failure (e.g., "green cover is too low"), you MUST modify the new prompt to fix that specific issue.
                You MUST respond by calling the `SimpleDesign` tool.
                """,
            ),
            ("human", "User Request: {user_request}\n\nRetrieved Compliance Rules:\n---\n{rag_context}\n---\n\nCritique/Feedback from Previous Attempt: {critique_feedback}"),
        ]
    )
    # Bind the Pydantic model as a tool to force the LLM to respond in the correct format
    planner_llm = llm.bind_tools(tools=[SimpleDesign], tool_choice="SimpleDesign")
    # Create the runnable chain that includes the prompt, the LLM, and the parser
    planner_runnable = prompt | planner_llm | PydanticToolsParser(tools=[SimpleDesign])
    return planner_runnable

def create_critique_agent():
    """
    Creates the Critique agent.
    This agent acts as a compliance officer, evaluating the quantitative analysis of a design
    against the retrieved compliance rules.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict compliance officer. Your task is to evaluate a design's analysis report against compliance rules.
                1.  **Analyze**: Compare the `analysis_results` (e.g., "green_cover_percentage": 15.2) with the `rag_context` (e.g., "green cover must be at least 20%").
                2.  **Decide**: If ALL rules are met, respond with the single word "PASS".
                3.  **Feedback**: If ANY rule is violated, respond with "FAIL", followed by a concise, one-sentence feedback for the Planner on what to fix.
                """,
            ),
            ("human", "Analysis Results:\n---\n{analysis_results}\n---\n\nCompliance Rules:\n---\n{rag_context}\n---"),
        ]
    )
    critique_runnable = prompt | llm
    return critique_runnable

def create_report_agent():
    """
    Creates the Report agent.
    This agent acts as a project manager, synthesizing the final successful results
    into a polished, human-readable markdown report.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a project manager. Your task is to create a final summary report for a successful design project in markdown format.
                The report must include these sections:
                1.  **Project Title**: A creative title.
                2.  **Initial Brief**: A summary of the original user request.
                3.  **Final Design Image**: A placeholder formatted as: `![Final Design](IMAGE_PATH_PLACEHOLDER)`.
                4.  **Compliance Analysis**: A summary of the final, compliant metrics.
                5.  **Conclusion**: A brief closing statement about the successful design.
                """,
            ),
            ("human", "User Request:\n---\n{user_request}\n---\n\nFinal Analysis Results:\n---\n{analysis_results}\n---"),
        ]
    )
    report_runnable = prompt | llm
    return report_runnable