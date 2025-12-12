"""
Main entry point for the lit_rag_ds project
"""
from litragds.mymod1 import example_function1, ExampleClass1

from litragds.mymod2 import example_function2


def main():
    """Main function for the application"""
    print("Welcome to lit_rag_ds - A lit RAG data science project!")
    
    # Using the function and class from litragds.example_module
    result = example_function1()
    print(f"Function1 result: {result}")
    
    result = example_function2()
    print(f"Function2 result: {result}")

    example_obj = ExampleClass1("ExampleInstance")
    greeting = example_obj.example_method()
    print(f"ExampleClass1 example_method result: {greeting}")
    
    
if __name__ == "__main__":
    main()