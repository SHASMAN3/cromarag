# import base64
# import io
# from PIL import Image
# import fitz
# from typing import List, Dict, Any
# import logging
# from .exceptions import ImageProcessingError
# from .utils import ProcessingConfig

# logger = logging.getLogger(__name__)

# class ImageProcessor:
#     def __init__(self, config: ProcessingConfig):
#         self.config = config

#     def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
#         """Extract images from PDF"""
#         images = []
#         try:
#             doc = fitz.open(pdf_path)
            
#             for page_num, page in enumerate(doc):
#                 for img_index, img in enumerate(page.get_images(full=True)):
#                     try:
#                         xref = img[0]
#                         base_image = doc.extract_image(xref)
#                         image_bytes = base_image['image']
                        
#                         # Verify image data is valid
#                         Image.open(io.BytesIO(image_bytes))
#                         images.append({
#                             'image': base64.b64encode(image_bytes).decode(),
#                             'page': page_num + 1
#                         })
#                     except Exception as e:
#                         logger.error(f"Error processing image on page {page_num + 1}: {e}")
            
#             return images
#         except Exception as e:
#             raise ImageProcessingError(f"Error extracting images: {e}")
#         finally:
#             if 'doc' in locals():
#                 doc.close()

#     def prepare_vision_prompt(self, query: str, text_context: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Prepare multimodal prompt with text and images"""
#         prompt_parts = [{
#             "text": f"""Based on the provided context, answer the following question.
#                 If you refer to specific content, include page numbers.
#                 If the information isn't in the context, say so clearly.

#                 Context:
#                 {text_context}

#                 Question: {query}

#                 Please provide a clear and concise answer based only on the given context.
#             """
#         }]
        
#         # Add images in the correct format
#         for img_data in images[:self.config.max_images]:
#             try:
#                 image_bytes = base64.b64decode(img_data['image'])
#                 prompt_parts.append({
#                     "inline_data": {
#                         "mime_type": "image/jpeg",
#                         "data": base64.b64encode(image_bytes).decode()
#                     }
#                 })
#             except Exception as e:
#                 logger.error(f"Error preparing image for prompt: {e}")
        
#         return prompt_parts


import base64
import io
import os
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from .exceptions import ImageProcessingError
from .utils import ProcessingConfig

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_image_config()

    def _setup_image_config(self):
        """Setup image processing configuration"""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.max_image_size = (1920, 1080)  # Max dimensions for processed images
        self.quality = 85  # JPEG quality for saved images
        
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate image dimensions and format"""
        try:
            if not image or image.size[0] <= 0 or image.size[1] <= 0:
                return False
            return True
        except Exception:
            return False

    def _process_image(self, image: Image.Image, enhance: bool = False) -> Image.Image:
        """Process image with optional enhancements"""
        try:
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Resize if larger than max dimensions
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            if enhance:
                # Apply basic enhancements
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                # Apply subtle noise reduction
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return image

    def _image_to_bytes(self, image: Image.Image, format: str = 'JPEG') -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format=format, quality=self.quality)
        return img_byte_array.getvalue()

    def extract_images(self, pdf_path: str, enhance: bool = False) -> List[Dict[str, Any]]:
        """Extract and process images from PDF"""
        images = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text blocks for OCR reference
                text_blocks = page.get_text("blocks")
                
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image['image']
                        
                        # Convert to PIL Image for processing
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        if self._validate_image(pil_image):
                            # Process the image
                            processed_image = self._process_image(pil_image, enhance)
                            
                            # Convert back to bytes and encode
                            processed_bytes = self._image_to_bytes(processed_image)
                            
                            # Get image location on page
                            image_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                            
                            images.append({
                                'image': base64.b64encode(processed_bytes).decode(),
                                'page': page_num + 1,
                                'format': base_image.get('ext', 'jpeg'),
                                'size': processed_image.size,
                                'location': image_rect,
                                'ocr_text': self._get_nearby_text(text_blocks, image_rect) if image_rect else ""
                            })
                    except Exception as e:
                        logger.error(f"Error processing image on page {page_num + 1}: {e}")
            
            return images
        except Exception as e:
            raise ImageProcessingError(f"Error extracting images: {e}")
        finally:
            if 'doc' in locals():
                doc.close()

    def _get_nearby_text(self, text_blocks: List[tuple], image_rect: fitz.Rect) -> str:
        """Get text near the image location"""
        if not image_rect:
            return ""
        
        nearby_text = []
        for block in text_blocks:
            block_rect = fitz.Rect(block[:4])
            # Check if block is near the image
            if block_rect.intersect(image_rect.expand(72)):  # 72 points = ~1 inch
                nearby_text.append(block[4])
        
        return " ".join(nearby_text)

    def process_input_image(self, image_path: str) -> Dict[str, Any]:
        """Process input image file"""
        try:
            with Image.open(image_path) as img:
                if not self._validate_image(img):
                    raise ImageProcessingError("Invalid image file")
                
                processed_image = self._process_image(img, enhance=True)
                image_bytes = self._image_to_bytes(processed_image)
                
                return {
                    'image': base64.b64encode(image_bytes).decode(),
                    'format': img.format.lower(),
                    'size': processed_image.size
                }
        except Exception as e:
            raise ImageProcessingError(f"Error processing input image: {e}")

    def prepare_vision_prompt(self, query: str, text_context: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare multimodal prompt with text and images"""
        prompt_parts = [{
            "text": f"""Based on the provided context and images, answer the following question.
                If you refer to specific content or images, include page numbers and image locations.
                If the information isn't in the context or images, say so clearly.

                Context:
                {text_context}

                Related Image Text:
                {' '.join(img.get('ocr_text', '') for img in images[:self.config.max_images])}

                Question: {query}

                Please provide a detailed answer based on both the text context and visual information.
            """
        }]
        
        # Add images in the correct format
        for img_data in images[:self.config.max_images]:
            try:
                image_bytes = base64.b64decode(img_data['image'])
                prompt_parts.append({
                    "inline_data": {
                        "mime_type": f"image/{img_data.get('format', 'jpeg')}",
                        "data": base64.b64encode(image_bytes).decode()
                    }
                })
            except Exception as e:
                logger.error(f"Error preparing image for prompt: {e}")
        
        return prompt_parts

    def save_processed_image(self, image_data: Dict[str, Any], output_path: str) -> str:
        """Save processed image to file"""
        try:
            image_bytes = base64.b64decode(image_data['image'])
            img = Image.open(io.BytesIO(image_bytes))
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with original format if supported, otherwise default to JPEG
            format = image_data.get('format', 'jpeg').upper()
            if format not in {'JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP'}:
                format = 'JPEG'
            
            img.save(output_path, format=format, quality=self.quality)
            return output_path
        except Exception as e:
            raise ImageProcessingError(f"Error saving processed image: {e}")