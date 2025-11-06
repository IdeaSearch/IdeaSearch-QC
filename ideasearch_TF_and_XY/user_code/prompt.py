# prompt.py - Prompt sections for IdeaSearch

prologue_section = r"""</think>The user's requirement is to generate a suitable quantum circuit ansatz with as few parameters as possible, so that it can be easily extended to multi-qubit systems while maintaining effectiveness<We are trying to find the most suitable quantum circuit for preparing the initial state of simulated gauge fields by generating an ansatz and performing variational optimization on it>

Function framework to implement:
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter,ParameterVector
def create_ansatz(n):
	# Ansatz implementation should go here
"""

epilogue_section = r"""
# Generation requirements:
# 1. Must correctly define the generation function
# 2. Function output should be physically reasonable (entanglement, rotation, etc.)
# 3. Avoid directly copying examples, complete at least one of the following (generalize previous code, simplify previous code, evolve previous code, create unique implementation)
# 4. Ensure code includes necessary mathematical operation library imports
# 5. Pay attention to code parameter simplicity and extensibility
# 6. Wrap the final code in a Python code block
</answer>"""
