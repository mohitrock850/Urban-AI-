import streamlit as st
import os
import time
from PIL import Image

# --- Load Environment Variables FIRST ---
# This ensures all API keys and LangSmith settings are loaded before other imports
from dotenv import load_dotenv
load_dotenv()

# --- Import Core Project Components ---
from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.agents import create_planner_agent, create_critique_agent, create_report_agent
from src.tools.rag_tool import rag_compliance_lookup
from src.tools.design_tool import generate_aerial_design
from src.tools.vision_tool import yolo_site_analyzer

# --- LangGraph Workflow Definition ---

# Define the nodes for the graph
def planner_node(state: GraphState) -> dict:
    planner = create_planner_agent()
    feedback = state.get("critique_feedback", "N/A")
    if state.get("human_approval") == "no":
        feedback += " The previous visual design was rejected. Please generate a significantly different design."

    planner_output = planner.invoke({
        "user_request": state["user_request"],
        "rag_context": state["rag_context"],
        "critique_feedback": feedback
    })
    design = planner_output[0]
    return {
        "dalle_prompt": design.dalle_prompt,
        "iteration_count": state["iteration_count"] + 1,
        "human_approval": None
    }

def designer_node(state: GraphState) -> dict:
    image_path = generate_aerial_design.invoke(state["dalle_prompt"])
    return {"image_path": image_path}

def analyst_node(state: GraphState) -> dict:
    analysis = yolo_site_analyzer.invoke(state["image_path"])
    return {"analysis_results": analysis}

def critique_node(state: GraphState) -> dict:
    critique_agent = create_critique_agent()
    critique_text = critique_agent.invoke({
        "analysis_results": state["analysis_results"],
        "rag_context": state["rag_context"]
    }).content
    
    if "FAIL" in critique_text.upper():
        return {"critique_feedback": critique_text}
    else:
        return {"critique_feedback": "PASS"}

def report_node(state: GraphState) -> dict:
    report_agent = create_report_agent()
    report_markdown = report_agent.invoke({
        "user_request": state["user_request"],
        "analysis_results": state["analysis_results"]
    }).content
    
    final_report = report_markdown.replace("IMAGE_PATH_PLACEHOLDER", str(state["image_path"]))
    return {"final_report": final_report}

# Compile the graph
@st.cache_resource
def compile_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("designer", designer_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("designer", "analyst")
    workflow.add_edge("analyst", "critique")
    
    def after_critique_router(state: GraphState) -> str:
        if state["critique_feedback"] == "PASS":
            return "report"
        else:
            if state["iteration_count"] >= 3:
                return END
            return "planner"

    workflow.add_conditional_edges("critique", after_critique_router, {"report": "report", "planner": "planner", END: END})
    workflow.add_edge("planner", "designer")
    workflow.add_edge("report", END)

    return workflow.compile()

# --- Streamlit UI ---

st.set_page_config(page_title="UrbanPlan AI", page_icon="🏙️", layout="wide")

st.title("🏙️ UrbanPlan AI")
st.markdown("A self-correcting, multi-modal planning agent that transforms design briefs into compliant urban prototypes.")

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'graph_state' not in st.session_state:
    st.session_state.graph_state = None


# Sidebar for inputs and controls
with st.sidebar:
    st.header("Design Brief")
    user_request = st.text_area(
        "Enter your high-level design goal:", 
        "A small building in a large green park.",
        height=150
    )
    
    start_button = st.button("Generate Design", type="primary", disabled=st.session_state.running)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Workflow")
    workflow_placeholder = st.empty()

with col2:
    st.subheader("Latest Design")
    image_placeholder = st.empty()


if start_button:
    st.session_state.running = True
    st.session_state.final_report = None
    
    with workflow_placeholder.status("📚 Looking up compliance rules...", expanded=True) as status:
        rag_context = rag_compliance_lookup.invoke(user_request)
        st.write("Retrieved relevant rules from the Master Plan for Delhi and compliance JSON.")
        status.update(label="✅ Compliance rules retrieved.", state="complete")
    
    initial_state = {
        "user_request": user_request,
        "rag_context": rag_context,
        "iteration_count": 0,
    }

    app = compile_graph()
    
    for output in app.stream(initial_state, {"recursion_limit": 5}):
        for key, value in output.items():
            st.session_state.graph_state = value
            
            with workflow_placeholder.status(f"Running Agent: **{key.upper()}**", expanded=True) as status:
                if key == "planner":
                    status.write("Drafting a new design plan...")
                elif key == "designer":
                    status.write("Generating a visual site plan with DALL-E 3...")
                    image = Image.open(value['image_path'])
                    image_placeholder.image(image, caption="Generated Site Plan", use_column_width=True)
                elif key == "analyst":
                    status.write("Analyzing the design with the custom-trained YOLOv8 model...")
                    st.write(f"**Analysis Results:**")
                    st.json(value['analysis_results'])
                elif key == "critique":
                    status.write("Evaluating design compliance...")
                    feedback = value['critique_feedback']
                    if feedback == "PASS":
                        st.success("✅ Design is compliant.")
                        status.update(label="✅ Critique: PASS", state="complete")
                    else:
                        st.warning(f"⚠️ Design is not compliant. Feedback: {feedback}")
                        status.update(label=f"⚠️ Critique: FAIL", state="running")
                        time.sleep(2)
                elif key == "report":
                    status.write("Generating the final project report...")
                    st.session_state.final_report = value.get("final_report")

    st.session_state.running = False
    st.rerun()

if st.session_state.final_report:
    st.subheader("Final Project Report")
    st.markdown(st.session_state.final_report)
    
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and st.session_state.graph_state:
        st.success("Workflow complete. You can view the full trace of this run in LangSmith.")

if not st.session_state.running and st.session_state.final_report:
    if st.button("Start New Design"):
        st.session_state.final_report = None
        st.session_state.graph_state = None
        st.rerun()

