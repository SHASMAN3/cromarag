import os
import gradio as gr
from main import create_gradio_interface

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Create and launch the interface
demo = create_gradio_interface()

# Configure for Hugging Face Spaces
demo.launch()