# initial_ideas.py - Initial seed ideas for IdeaSearch

initial_ideas = [
    # Template 1: Simple RX rotation with linear parameter scaling
    """from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_ansatz(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('params', 2)
    for qubit in range(n):
        qc.rx(params[0] + params[1] * qubit / n, qubit)
    return qc""",
    
    # Template 2: RY rotation with grouped parameters
    """from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_ansatz(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('params', 3)
    for qubit in range(n):
        qc.ry(params[qubit // 3], qubit)
    return qc""",
    
    # Template 3: RZ rotation with exponential parameter
    """import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_ansatz(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('params', 2)
    for qubit in range(n):
        qc.rz(params[0] + np.exp(params[1]), qubit)
    return qc"""
]
