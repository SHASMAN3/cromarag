import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .exceptions import ModelInitializationError
from .utils import ProcessingConfig

class ModelManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_models()

    def setup_models(self):
        """Initialize AI models"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ModelInitializationError("GOOGLE_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=self.config.temperature
            )
            
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
        except Exception as e:
            raise ModelInitializationError(f"Error initializing models: {e}")

    def generate_text_response(self, prompt: str) -> str:
        """Generate text-only response"""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            raise ModelInitializationError(f"Error generating text response: {e}")

    def generate_multimodal_response(self, prompt_parts: list[dict[str, any]]) -> str:
        """Generate response with text and images"""
        try:
            response = self.vision_model.generate_content(prompt_parts)
            return response.text
        except Exception as e:
            raise ModelInitializationError(f"Error generating multimodal response: {e}")