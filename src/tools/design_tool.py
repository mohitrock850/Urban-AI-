# src/tools/design_tool.py

import os
from openai import OpenAI
import requests
from langchain.tools import tool
from datetime import datetime

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the directory to save generated images
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@tool
def generate_aerial_design(design_prompt: str) -> str:
    """
    Generates a detailed, top-down, architectural site plan diagram using DALL-E 3.
    """
    print("Generating design with DALL-E...")
    try:
        # Enhance the user's prompt with specific instructions for DALL-E
        full_dalle_prompt = (
            f"Highly detailed top-down architectural site plan diagram. "
            f"Focus on clear, distinct shapes for: buildings, green parks, roads, and water bodies. "
            f"Rendered in a clear, legible style suitable for computer vision analysis. "
            f"The design should feature: {design_prompt}"
        )
        
        # Call the DALL-E 3 API
        response = client.images.generate(
            model="dall-e-3", 
            prompt=full_dalle_prompt, 
            size="1024x1024", 
            quality="standard", 
            n=1
        )

        # Download the generated image from the URL provided by the API
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status() # Raise an error for bad status codes

        # Save the image to a file with a unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(OUTPUT_DIR, f"design_{timestamp}.png")
        
        with open(file_path, "wb") as f:
            f.write(image_response.content)
            
        print(f"Design saved to {file_path}")
        return file_path
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"