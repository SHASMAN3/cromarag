import fitz
from typing import Dict, Any
import logging
from .exceptions import DocumentProcessingError
from .utils import ProcessingConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF"""
        content = {
            'text': [],
            'tables': []
        }
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    content['text'].append({
                        'content': text,
                        'page': page_num + 1
                    })
                
                # Extract tables
                tables = page.find_tables()
                if tables:
                    for table in tables:
                        content['tables'].append({
                            'content': table.extract(),
                            'page': page_num + 1
                        })
            
            return content
        except Exception as e:
            raise DocumentProcessingError(f"Error extracting PDF content: {e}")
        finally:
            if 'doc' in locals():
                doc.close()

    def create_chunks(self, text_content: str) -> list[str]:
        """Split text into chunks for processing"""
        try:
            return self.text_splitter.split_text(text_content)
        except Exception as e:
            raise DocumentProcessingError(f"Error creating text chunks: {e}")