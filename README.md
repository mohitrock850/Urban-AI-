# 🏙️ UrbanPlan AI: A Self-Correcting, Multi-Modal Planning Agent

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-0.1-orange?logo=langchain&logoColor=white">
  <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Custom_Trained-blueviolet?logo=yolo">
  <img alt="LangSmith" src="https://img.shields.io/badge/Observability-LangSmith-black?logo=langsmith">
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-GPT_4o_&_DALL·E_3-00A67E?logo=openai&logoColor=white">
</p>

---

## 🧩 Overview

**UrbanPlan AI** is an advanced, end-to-end multi-agent system demonstrating skills in **agentic workflows**, **custom computer vision**, and **self-correcting AI**.

It translates **high-level natural language design briefs** into **visually realized, regulation-compliant urban prototypes**.  
The system uses an **agentic workflow** to design, analyze, critique, and self-correct until the generated plan satisfies all compliance rules sourced from **real-world planning documents**.

This project bridges the gap between **language (the brief)** and **vision (the site plan)** using a **custom-trained YOLOv8 model** for quantitative analysis, overcoming the limitations of standard LLMs.

---

## 🌐 Live Demo & Observability

- **Live App:** [Your Streamlit Community Cloud URL]
- **AI Observability:** View the AI’s “Brain” on **LangSmith**  
  *(See every thought, tool call, and decision the AI makes in real time.)*

---

## 🏆 Key Skills & Concepts Demonstrated

### 🧠 AI Engineering

- **Multi-Agent Systems:**  
  Designing and orchestrating a team of specialized AI agents (Planner, Designer, Analyst, Critique, Report).

- **Agentic Workflows:**  
  Building a **stateful, cyclical graph (LangGraph)** for looping and self-correction based on feedback.

- **Tool Use & Function Calling:**  
  Creating custom tools for the AI (RAG, Vision) and enforcing tool use for reliable outputs.

- **Observability & Debugging:**  
  Using **LangSmith** for tracing and analyzing complex agent behaviors to ensure production readiness.

---

### 👁️ Computer Vision (MLOps)

- **Custom Fine-Tuning:**  
  Training a specialized **YOLOv8 model** from scratch on a synthetic dataset.

- **Data Augmentation:**  
  Expanding a small dataset programmatically (via **Google Colab**) for robustness.

- **Data Annotation:**  
  Labeling custom datasets using **Roboflow** for architectural plan detection.

---

### 🎨 Generative AI

- **Multi-Modality:**  
  Integrating **LLMs (GPT-4o)** with **diffusion models (DALL·E 3)** and **Computer Vision**.

- **Retrieval-Augmented Generation (RAG):**  
  Using **ChromaDB** to store and retrieve factual data from PDFs (e.g., *Delhi Master Plan*) and JSONs, ensuring grounded outputs.

---

### 💻 Software Engineering

- **Full-Stack Application:**  
  Built with **Streamlit** frontend and **modular Python backend**.

- **Maintainable Architecture:**  
  - `streamlit_app.py` → UI  
  - `state.py` → Stateful logic  
  - `agents.py` → Agent definitions  
  - `tools/` → Custom tools (RAG, Vision, etc.)

---

## 🛠️ Technologies & Tools Used

| **Category** | **Technology** |
|---------------|----------------|
| AI Orchestration | LangGraph, LangChain |
| AI Observability | LangSmith |
| Language Models | OpenAI GPT-4o |
| Image Generation | DALL·E 3 |
| Computer Vision | YOLOv8, Ultralytics, OpenCV |
| Training Platform | Google Colab |
| Web Framework | Streamlit |
| Vector Database | ChromaDB |
| Data Parsing | PyPDFLoader, JSONLoader, Pydantic |
| Core Stack | Python |

---

## 🧠 Application Logic Flowchart

The system's “brain” is a **stateful graph** orchestrated by **LangGraph**, enabling self-correcting AI behavior.

```mermaid
graph TD
    A[Start: User Enters Brief in Streamlit] --> B[RAG: 'rag_compliance_lookup' <br> (Finds rules in Delhi PDF & JSON)];
    B --> C[Planner Agent: 'create_planner_agent' <br> (Generates DALL-E prompt based on rules & feedback)];
    C --> D[Designer Tool: 'generate_aerial_design' <br> (Creates site plan image)];
    D --> E[Analyst Tool: 'yolo_site_analyzer' <br> (Custom YOLOv8 model measures image)];
    E --> F[Critique Agent: 'create_critique_agent' <br> (Compares YOLO results to RAG rules)];
    F --> G{"Compliance Check"};
    G -- FAIL (Loop < 3) --> H[Feedback Sent to Planner];
    H --> C;
    G -- PASS --> I[Report Agent: 'create_report_agent' <br> (Generates final summary)];
    I --> J[End: Display Report in Streamlit];
    G -- FAIL (Loop >= 3) --> K[End: Max Iterations Reached];

    style A fill:#D6EAF8,stroke:#3498DB
    style J fill:#D5F5E3,stroke:#2ECC71
    style K fill:#FADBD8,stroke:#E74C3C
```

🖥️ Interface

The application is presented in a Streamlit dashboard.

Left Sidebar

User inputs a design brief (e.g., “A small park with 20% green space”).

Main View

Displays the live status of agents (Planner, Designer, Analyst, Critique).

Shows the generated design image for visual inspection.

Final Report

Once a design passes compliance checks, a summary report is generated.

<p align="center"> <img src="https://your-image-url" alt="Streamlit UI Screenshot" width="800"> </p>
🏃‍♂️ Run Locally

1️⃣ Prerequisites

Custom AI Model (Required)
This project requires a custom-trained YOLOv8 model.

Train your model (e.g., on Google Colab).

Place your trained model file in the /models directory.

Ensure the file is named:

urbanplan_yolov8.pt

2️⃣ Clone Repository
git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPO-NAME].git
cd [YOUR-REPO-NAME]

3️⃣ Create Virtual Environment
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Configure Environment

Create a .env file in the project root and add your API keys:

# .env

# OpenAI API Key
OPENAI_API_KEY="sk-..."

# LangSmith Observability Settings
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="UrbanPlan AI"

6️⃣ Run the Application

When running for the first time, the rag_tool.py will automatically build the ChromaDB vector store from your data files (may take a few minutes).

streamlit run streamlit_app.py


Then open your browser to:

👉 http://localhost:8501

<p align="center"> <b>💡 UrbanPlan AI — Turning ideas into verified designs.</b> </p> ```
