# test_tool.py (Reverted to a simpler version)

import os
from dotenv import load_dotenv

load_dotenv()

from src.tools.rag_tool import rag_compliance_lookup
from src.agents import create_planner_agent
from src.tools.design_tool import generate_aerial_design

def main():
    """Main function to test the simplified RAG -> Planner -> Designer flow."""

    user_request = "Design a mixed-use development with a central park, residential towers on the north side, and low-rise commercial buildings on the south."
    
    print(f"--- 🚀 Starting test with request: '{user_request}' ---")

    print("\n--- 📚 Performing RAG Lookup ---")
    rag_context = rag_compliance_lookup.invoke(user_request)
    print("✅ RAG context retrieved.")

    print("\n--- 🧠 Invoking Planner Agent ---")
    planner_agent = create_planner_agent()
    
    planner_output = planner_agent.invoke({
        "user_request": user_request,
        "rag_context": rag_context,
    })
    
    # Extract the single object from the list returned by the parser
    simple_design = planner_output[0] 
    
    print("✅ Planner agent created a DALL-E prompt:")
    print(f"   '{simple_design.dalle_prompt}'")

    print("\n--- 🎨 Invoking Designer Tool ---")
    image_path = generate_aerial_design.invoke(simple_design.dalle_prompt)
    if "error" in image_path:
        print(f"❌ Designer tool failed: {image_path}")
        return
        
    print(f"✅ Designer tool created image at: {image_path}")
    print("\n--- 🎉 Test complete! ---")


if __name__ == "__main__":
    main()
