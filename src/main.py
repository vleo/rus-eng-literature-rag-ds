"""
Main entry point for the lit_rag_ds project
"""
from litragds.example_module import example_function, ExampleClass


def main():
    """Main function for the application"""
    print("Welcome to lit_rag_ds - A lit RAG data science project!")
    
    # Using the function and class from litragds.example_module
    result = example_function()
    print(f"Function result: {result}")
    
    example_obj = ExampleClass("ExampleInstance")
    greeting = example_obj.greet()
    print(f"Class method result: {greeting}")
    
    
if __name__ == "__main__":
    main()