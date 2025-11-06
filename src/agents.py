# src/agents.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# --- Planner Pydantic Model ---
class SimpleDesign(BaseModel):
    """A model to hold the DALL-E prompt for the site plan."""
    dalle_prompt: str = Field(description="A detailed DALL-E 3 prompt for generating a top-down, architectural site plan diagram.")

# --- Agent Definitions ---

def create_planner_agent():
    """Creates the Planner agent that generates a DALL-E prompt based on rules and feedback."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert urban planner. Your task is to convert a user's request, compliance rules, and critique feedback into a single, detailed DALL-E prompt.
                
                **Crucially, if you receive feedback about a compliance failure (e.g., "green cover is too low"), you MUST dramatically alter the new prompt to fix that specific issue.** For example, if green space is too low, use phrases like "a vast central park," "extensive green spaces," "covered in lush greenery," or "buildings are secondary to the large park."
                
                You MUST respond by calling the `SimpleDesign` tool.
                """,
            ),
            ("human", "User Request: {user_request}\n\nRetrieved Compliance Rules:\n---\n{rag_context}\n---\n\nCritique/Feedback from Previous Attempt: {critique_feedback}"),
        ]
    )
    planner_llm = llm.bind_tools(tools=[SimpleDesign], tool_choice="SimpleDesign")
    planner_runnable = prompt | planner_llm | PydanticToolsParser(tools=[SimpleDesign])
    return planner_runnable

def create_critique_agent():
    """
    Creates the Critique agent.
    This agent's job is now to compare the analysis results against
    a fixed set of rules.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict compliance officer. Your task is to evaluate a design's analysis report.

                **Compliance Rules:**
                1.  `green_cover_percentage` must be >= 15
                2.  `building_footprint_percentage` must be <= 40

                **Your Task:**
                1.  **Analyze**: Look at the `analysis_results` (e.g., `{'green_cover_percentage': 21.55, ...}`).
                2.  **Compare**: Check if the results meet ALL the compliance rules defined above.
                3.  **Decide**: If ALL rules are met, respond with the single word "PASS".
                4.  **Feedback**: If ANY rule is violated, respond with "FAIL", followed by a concise, one-sentence feedback for the Planner on what to fix.
                """,
            ),
            # This prompt ONLY expects the 'analysis_results' variable
            ("human", "Analysis Results:\n---\n{analysis_results}\n---"),
        ]
    )
    critique_runnable = prompt | llm
    return critique_runnable

def create_report_agent():
    """Creates the Report agent that synthesizes the final results into a markdown report."""
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
                5.  **Conclusion**: A brief closing statement.
                """,
            ),
            ("human", "User Request:\n---\n{user_request}\n---\n\nFinal Analysis Results:\n---\n{analysis_results}\n---"),
        ]
    )
    report_runnable = prompt | llm
    return report_runnable