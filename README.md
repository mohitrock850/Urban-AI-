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

**UrbanPlan AI** is an advanced **multi-agent system** that translates natural language **design briefs** into **visually realized, regulation-compliant urban prototypes**.  

It uses an **agentic workflow** to autonomously **design**, **analyze**, **critique**, and **self-correct** its work by comparing a **custom-trained YOLOv8 model’s** analysis against a **RAG knowledge base** of real-world planning documents.

---

## 🚀 Live Demo: The Self-Correction Loop

The app’s core feature is its ability to **identify its own failures and iterate until it succeeds**.

1. 🧾 User Enters Brief  
2. ❌ AI Fails, Critiques, & Tries Again  
3. ✅ AI Succeeds & Reports  

<p align="center">
  <img src="./images/main.png" alt="User Enters Prompt" width="400">
  <img src="./images/demo_fail.jpg" alt="AI Fails Compliance Check" width="400">
  <img src="./images/demo_pass.jpg" alt="AI Passes Compliance Check" width="400">
</p>

---

## 🧠 Core Architecture & Logic

The system’s “brain” is a **stateful graph orchestrated by LangGraph**, enabling **complex, cyclical AI behavior**.

```mermaid
graph TD
    A[Start: User Enters Brief] --> B[RAG: Find Rules in PDF/JSON];
    B --> C[Planner Agent: Generate DALL-E Prompt];
    C --> D[Designer Tool: Create Site Plan Image];
    D --> E[Analyst Tool: Custom YOLOv8 Measures Image];
    E --> F[Critique Agent: Compare YOLO results to RAG Rules];
    F --> G{"Compliance Check"};
    G -- FAIL (Loop < 3) --> H[Feedback Sent to Planner];
    H --> C;
    G -- PASS --> I[Report Agent: Generate Final Summary];
    I --> J[End: Display Report];
    G -- FAIL (Loop >= 3) --> K[End: Max Iterations Reached];

    style A fill:#D6EAF8,stroke:#3498DB
    style J fill:#D5F5E3,stroke:#2ECC71
    style K fill:#FADBD8,stroke:#E74C3C
```
## 🏆 Key Skills & Concepts Demonstrated

| **Skill** | **Description** |
|------------|-----------------|
| 🤖 **Multi-Agent Systems** | Orchestrated a team of specialized AI agents (Planner, Critique, etc.). |
| 🔄 **Agentic Workflows** | Built a stateful, self-correcting loop using LangGraph. |
| 🛠️ **Tool Use** | Enabled agents to reliably use custom-built tools (Vision, RAG). |
| 👁️ **Computer Vision (MLOps)** | Fine-tuned a YOLOv8 model on a custom dataset for domain-specific analysis. |
| 🎨 **Multi-Modality** | Bridged NLP (GPT-4o), Image Gen (DALL·E 3), and CV (YOLOv8). |
| 📚 **RAG** | Grounded AI in facts using a ChromaDB vector store of PDF/JSON documents. |
| 💻 **Full-Stack App** | Developed a complete Streamlit frontend with a modular Python backend. |
| 🧠 **Observability** | Integrated LangSmith to trace and debug complex agent behavior. |

---

## 🛠️ Tech Stack

| **Category** | **Technology** |
|---------------|----------------|
| **AI Orchestration** | <img src="https://img.shields.io/badge/LangGraph-orange?logo=langchain&logoColor=white"> <img src="https://img.shields.io/badge/LangChain-white?logo=langchain&logoColor=black"> |
| **AI Observability** | <img src="https://img.shields.io/badge/LangSmith-black?logo=langsmith&logoColor=white"> |
| **Language & Vision** | <img src="https://img.shields.io/badge/OpenAI-GPT_4o_&_DALL·E_3-00A67E?logo=openai&logoColor=white"> |
| **Computer Vision** | <img src="https://img.shields.io/badge/YOLOv8-blueviolet?logo=yolo"> <img src="https://img.shields.io/badge/OpenCV-blue?logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/Google_Colab-orange?logo=googlecolab&logoColor=white"> |
| **Web Framework** | <img src="https://img.shields.io/badge/Streamlit-red?logo=streamlit&logoColor=white"> |
| **Vector Database** | <img src="https://img.shields.io/badge/ChromaDB-blue"> |
| **Core Stack** | <img src="https://img.shields.io/badge/Python-blue?logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Pydantic-e92063?logo=pydantic&logoColor=white"> |

---
## 🎨 Sample Design Gallery

Examples of complex designs generated and analyzed by the system:

| **Sample 1** | **Sample 2** |
|---------------|--------------|
| <img src="./images/sample.jpg" alt="Sample Design 1" width="400"> | <img src="./images/sample2.jpg" alt="Sample Design 2" width="400"> |

---

## 🏃‍♂️ Run Locally

### 1️⃣ Prerequisites

This project requires the custom-trained YOLOv8 model file:

```bash
/models/urbanplan_yolov8.pt
```

### 2️⃣ Clone Repository

```bash
git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPO-NAME].git
cd [YOUR-REPO-NAME]
```
### 3️⃣ Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```
### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
### 5️⃣ Configure Environment
Create a .env file in the project root:
```bash
# .env
OPENAI_API_KEY="sk-..."
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="UrbanPlan AI"
```
### 6️⃣ Run the Application 

The first run will take a minute to build the ChromaDB vector store.

```bash
streamlit run streamlit_app.py
```
Then open your browser to:

👉 http://localhost:8501

<p align="center"> <b>💡 UrbanPlan AI — Turning ideas into verified designs.</b> </p> 
