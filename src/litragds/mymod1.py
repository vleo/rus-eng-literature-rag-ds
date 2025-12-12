"""
Example module for the litragds package
"""

from litragds import example_function2, ExampleClass2

print("I'm in mymod1")

def example_function1():
    """
    An example function in the litragds package
    """
    return "This is an example function from the litragds module!"

class ExampleClass1:
    """
    An example class in the litragds package
    """
    
    def __init__(self, name: str):
        self.name = name
        self.c2 = ExampleClass2("xyzzy")
        
    def example_method(self):
        x = example_function2()
        y = self.c2.example_method()
        
        return f"Hello from {self.name} and {x} and {y}!"