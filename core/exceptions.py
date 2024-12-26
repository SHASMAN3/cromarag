class DocumentProcessingError(Exception):
    """Raised when there's an error processing a document"""
    pass

class ModelInitializationError(Exception):
    """Raised when there's an error initializing AI models"""
    pass

class ImageProcessingError(Exception):
    """Raised when there's an error processing images"""
    pass

class RetrieverError(Exception):
    """Raised when there's an error with the retrieval system"""
    pass