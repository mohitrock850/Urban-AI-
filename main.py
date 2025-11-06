# --- Load Environment Variables FIRST ---
from dotenv import load_dotenv
load_dotenv()

# --- Now, import everything else ---
import os
from langgraph.graph import StateGraph, END
from datetime import datetime
from src.state import GraphState
from src.agents import create_planner_agent, create_critique_agent, create_report_agent
from src.tools.rag_tool import rag_compliance_lookup
from src.tools.design_tool import generate_aerial_design
from src.tools.vision_tool import yolo_site_analyzer


# --- Define Graph Nodes ---
def planner_node(state: GraphState) -> dict:
    print("\n--- 🧠 PLANNER ---")
    planner = create_planner_agent()
    feedback = state.get("critique_feedback", "N/A")
    if state.get("human_approval") == "no":
        feedback += " The previous visual design was rejected by the user. Please generate a significantly different design."

    planner_output = planner.invoke({
        "user_request": state["user_request"],
        "rag_context": state["rag_context"],
        "critique_feedback": feedback
    })
    design = planner_output[0]
    return {
        "dalle_prompt": design.dalle_prompt,
        "iteration_count": state["iteration_count"] + 1,
        "human_approval": None # Reset human approval
    }

def designer_node(state: GraphState) -> dict:
    print("\n--- 🎨 DESIGNER ---")
    image_path = generate_aerial_design.invoke(state["dalle_prompt"])
    return {"image_path": image_path}

def human_in_the_loop_node(state: GraphState) -> dict:
    print("\n--- 🧑‍⚖️ HUMAN APPROVAL ---")
    print(f"Design image generated at: {state['image_path']}")
    user_input = ""
    while user_input.lower() not in ["yes", "no"]:
        user_input = input("Approve the visual design? (yes/no): ")
    return {"human_approval": user_input.lower()}

def analyst_node(state: GraphState) -> dict:
    print("\n--- 👁️ ANALYST ---")
    analysis = yolo_site_analyzer.invoke(state["image_path"])
    return {"analysis_results": analysis}

def critique_node(state: GraphState) -> dict:
    print("\n--- 🧐 CRITIQUE ---")
    critique_agent = create_critique_agent()
    critique_text = critique_agent.invoke({
        "analysis_results": state["analysis_results"],
        "rag_context": state["rag_context"]
    }).content
    print(f"Critique: {critique_text}")
    if "FAIL" in critique_text.upper():
        return {"critique_feedback": critique_text}
    else:
        return {"critique_feedback": "PASS"}

def report_node(state: GraphState) -> dict:
    print("\n--- 📝 REPORT ---")
    report_agent = create_report_agent()
    report_markdown = report_agent.invoke({
        "user_request": state["user_request"],
        "analysis_results": state["analysis_results"]
    }).content
    
    final_report = report_markdown.replace("IMAGE_PATH_PLACEHOLDER", str(state["image_path"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"./outputs/report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(final_report)
        
    print(f"Final report saved to {report_path}")
    print("\n--- FINAL REPORT ---")
    print(final_report)
    return {}

# --- Define Conditional Edge Logic ---
def after_critique_router(state: GraphState) -> str:
    print("\n--- ❓ COMPLIANCE CHECK ---")
    if state["critique_feedback"] == "PASS":
        print("Decision: Design is compliant. Proceeding to report.")
        return "report"
    else:
        if state["iteration_count"] >= 3:
            print("Decision: Max iterations reached. Ending workflow.")
            return "end"
        print("Decision: Design is not compliant. Looping back to Planner.")
        return "continue"

def after_hitl_router(state: GraphState) -> str:
    print("\n--- ❓ HUMAN DECISION ---")
    if state["human_approval"] == "yes":
        print("Decision: Human approved. Proceeding to analysis.")
        return "analyst"
    else:
        print("Decision: Human rejected. Returning to Planner.")
        return "planner"

# --- Create and Compile the Graph ---
def run_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("designer", designer_node)
    workflow.add_node("human_in_the_loop", human_in_the_loop_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "designer")
    workflow.add_edge("designer", "human_in_the_loop")
    workflow.add_conditional_edges(
        "human_in_the_loop", after_hitl_router, {"analyst": "analyst", "planner": "planner"}
    )
    workflow.add_edge("analyst", "critique")
    
    workflow.add_conditional_edges(
        "critique", after_critique_router, {"continue": "planner", "report": "report", "end": END}
    )
    
    workflow.add_edge("report", END)

    app = workflow.compile()

    # --- Run the graph ---
    user_request = "A small building in a large green park."
    rag_context = rag_compliance_lookup.invoke(user_request)
    inputs = {
        "user_request": user_request,
        "rag_context": rag_context,
        "iteration_count": 0,
    }
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' output received.")

if __name__ == "__main__":
    run_graph()