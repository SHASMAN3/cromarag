import os
import logging
import gradio as gr
from dotenv import load_dotenv
from core import (
    DocumentProcessor,
    ImageProcessor,
    ModelManager,
    Retriever,
    ProcessingConfig,
    setup_logging
)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class MultimodalRAG:
    def __init__(self):
        """Initialize the RAG system"""
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration and components
        self.config = ProcessingConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.img_processor = ImageProcessor(self.config)
        self.model_manager = ModelManager(self.config)
        self.retriever = Retriever(self.config)

    def process_document(self, pdf_path: str):
        """Process uploaded PDF document"""
        try:
            logger.info(f"Processing document: {pdf_path}")
            
            # Extract content from PDF
            text_content = self.doc_processor.extract_pdf_content(pdf_path)
            images = self.img_processor.extract_images(pdf_path)
            
            # Create text chunks 
            all_text = []
            for text_item in text_content['text']:
                all_text.append(f"[Page {text_item['page']}] {text_item['content']}")
            text_combined = "\n".join(all_text)
            chunks = self.doc_processor.create_chunks(text_combined)
            
            # Get embeddings and store in ChromaDB
            embeddings = self.model_manager.embeddings.embed_documents(chunks)
            self.retriever.reset()  # Reset for new document
            self.retriever.add_chunks(chunks, embeddings)
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks created")
            return {'text_content': text_content, 'images': images}
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    def generate_response(self, query: str, context):
        """Generate response to user query"""
        try:
            logger.info(f"Generating response for query: {query}")
            
            # Get query embedding and retrieve relevant chunks
            query_embedding = self.model_manager.embeddings.embed_query(query)
            relevant_chunks = self.retriever.retrieve_relevant(query, query_embedding)
            text_context = "\n".join(relevant_chunks)
            
            # Check if query is image-related
            image_related = any(word in query.lower() for word in 
                              ['image', 'figure', 'picture', 'diagram', 'graph', 'show', 'visual'])
            
            if image_related and context['images']:
                logger.info("Processing image-related query with multimodal model")
                prompt_parts = self.img_processor.prepare_vision_prompt(
                    query, text_context, context['images']
                )
                return self.model_manager.generate_multimodal_response(prompt_parts)
            else:
                logger.info("Processing text-only query")
                prompt = f"""Based on the provided context, answer the following question.
                    If you refer to specific content, include page numbers.
                    If the information isn't in the context, say so clearly.

                    Context:
                    {text_context}

                    Question: {query}

                    Please provide a clear and concise answer based only on the given context.
                    """
                return self.model_manager.generate_text_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try rephrasing your question."

def create_gradio_interface():
    """Create and configure Gradio interface"""
    rag = MultimodalRAG()
    
    def process_query(pdf_file: str, query: str) -> str:
        """Handle the query processing"""
        if not pdf_file or not query:
            return "Please provide both a PDF file and a query."
            
        try:
            content = rag.process_document(pdf_file)
            return rag.generate_response(query, content)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"An error occurred: {str(e)}"

    # Custom CSS for better UI
    custom_css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
        padding: 20px;
    }
    .input-section {
        margin-bottom: 20px;
    }
    .output-section {
        margin-top: 20px;
    }
    """

    # Create Gradio interface
    demo = gr.Interface(
        fn=process_query,
        inputs=[
            gr.File(
                label="Upload PDF Document",
                type="filepath",
                file_types=[".pdf"]
            ),
            gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the document...",
                lines=2
            )
        ],
        outputs=gr.Textbox(label="Answer", lines=5),
        title="📚 Advanced PDF Document Assistant",
        description="""Upload any PDF document and ask questions about its content. 
        This system can understand and answer questions about text, images, and tables within your document.""",
        css=custom_css,
        allow_flagging="never",
        theme="soft"
    )
    
    return demo

def main():
    """Main function to run the application"""
    try:
        logger.info("Starting the application")
        demo = create_gradio_interface()
        
        # Launch with custom configurations
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True
        )
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        raise

if __name__ == "__main__":
    main()