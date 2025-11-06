from typing import TypedDict, Optional

class GraphState(TypedDict):
    """
    Represents the state of our urban planning agent graph.
    """
    user_request: str
    rag_context: str
    dalle_prompt: Optional[str]
    image_path: Optional[str]
    analysis_results: Optional[dict]
    critique_feedback: Optional[str]
    human_approval: Optional[str]
    iteration_count: int
    final_report: Optional[str]
