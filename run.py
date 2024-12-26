from app.main import MultimodalRAG
from data_processor import DataOutputProcessor

def run_example():
    # Initialize the RAG system
    rag = MultimodalRAG()
    
    # Initialize the output processor
    processor = DataOutputProcessor()
    
    # Sample data to process
    sample_data = {
        'text': 'This is a sample document text.',
        'table_data': [
            {'Column1': 'Value1', 'Column2': 'Value2'},
            {'Column1': 'Value3', 'Column2': 'Value4'}
        ],
        'visualization_data': {
            'Category A': 10,
            'Category B': 25,
            'Category C': 15,
            'Category D': 30
        }
    }
    
    # Process the data
    visualization_config = {
        'title': 'Sample Data Distribution',
        'chart_type': 'bar'
    }
    
    # Generate outputs
    outputs = processor.process_retrieval_results(sample_data, visualization_config)
    
    # Print results
    print("\nGenerated outputs:")
    for output_type, path in outputs.items():
        print(f"{output_type}: {path}")
    
    # Run the Gradio interface
    print("\nStarting Gradio interface...")
    from main import create_gradio_interface
    demo = create_gradio_interface()
    demo.launch()

if __name__ == "__main__":
    run_example()
