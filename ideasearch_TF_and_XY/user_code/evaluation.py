# evaluation.py - Evaluation function for IdeaSearch with TF and XY Hamiltonians

import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import io
import traceback
import re
import threading


# Global variable to select model type: 'TF' or 'XY'
MODEL_TYPE = 'TF'  # Change this to 'XY' to use XY model


class TimeoutException(Exception):
    """Exception raised when evaluation times out"""
    pass


def extract_python_code(content: str) -> str:
    """
    Extract Python code from markdown code blocks or return content as-is.
    
    Args:
        content: String that may contain markdown code blocks
        
    Returns:
        Extracted Python code or original content
    """
    # Try to match Python code blocks with ```python or ```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # If no code blocks found, return original content
    return content.strip()


def generate_TF_hamiltonian(num_qubits, J=1.0, h=1.0, PBC=0):
    """
    Generate Transverse Field Ising Hamiltonian
    H = -J * Σ Z_i Z_{i+1} - h * Σ X_i
    """
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    
    # ZZ interaction terms and X field terms
    for i in range(num_qubits - 1): 
        zz_term = "I" * i + "ZZ" + "I" * (num_qubits - i - 2)
        terms.append(SparsePauliOp(zz_term, -J))
        
        x_term = "I" * i + "X" + "I" * (num_qubits - i - 1)
        terms.append(SparsePauliOp(x_term, -h))
    
    # Last X term
    x_term = "I" * (num_qubits - 1) + "X"
    terms.append(SparsePauliOp(x_term, -h))
    
    # Periodic boundary condition
    if PBC == 1:
        zz_term = "Z" + "I" * (num_qubits - 2) + "Z"
        terms.append(SparsePauliOp(zz_term, -J))
    
    hamiltonian = sum(terms)
    return hamiltonian


def generate_XY_hamiltonian(num_qubits, J=1.0, h=1.0, PBC=0):
    """
    Generate XY Model Hamiltonian
    H = -J * Σ (X_i X_{i+1} + Y_i Y_{i+1}) - h * Σ Z_i
    """
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    
    # XX and YY interaction terms
    for i in range(num_qubits - 1):
        # XX term
        xx_term = "I" * i + "XX" + "I" * (num_qubits - i - 2)
        terms.append(SparsePauliOp(xx_term, -J))
        
        # YY term
        yy_term = "I" * i + "YY" + "I" * (num_qubits - i - 2)
        terms.append(SparsePauliOp(yy_term, -J))
        
        # Z field term
        z_term = "I" * i + "Z" + "I" * (num_qubits - i - 1)
        terms.append(SparsePauliOp(z_term, -h))
    
    # Last Z term
    z_term = "I" * (num_qubits - 1) + "Z"
    terms.append(SparsePauliOp(z_term, -h))
    
    # Periodic boundary condition
    if PBC == 1:
        # XX term
        xx_term = "X" + "I" * (num_qubits - 2) + "X"
        terms.append(SparsePauliOp(xx_term, -J))
        
        # YY term
        yy_term = "Y" + "I" * (num_qubits - 2) + "Y"
        terms.append(SparsePauliOp(yy_term, -J))
    
    hamiltonian = sum(terms)
    return hamiltonian


def _evaluate_with_timeout(idea_content: str, model_type: str, timeout: float = 240.0):
    """
    Internal function to perform evaluation with timeout protection.
    
    Args:
        idea_content: String containing the code for create_ansatz function
        model_type: 'TF' or 'XY'
        timeout: Timeout in seconds (default 240)
        
    Returns:
        tuple: (score: float, info: str or None)
    """
    # System parameters
    N = 9       # Number of qubits
    J = 1.0     # Coupling strength
    h = 1.0     # Field strength
    PBC = 0     # 0 for open boundary, 1 for periodic boundary
    
    # Extract Python code from markdown blocks if present
    idea_content = extract_python_code(idea_content)
        
    # Import qiskit modules
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Estimator
    
    # Create namespace for exec with necessary imports
    namespace = {
        'np': np,
        'QuantumCircuit': QuantumCircuit,
        'Parameter': Parameter,
        'ParameterVector': ParameterVector,
    }
        
    # Execute the code to define create_ansatz function
    try:
        exec(idea_content, namespace)
    except Exception as e:
        return 0.0, f"Code execution error: {str(e)}"
    
    # Check if create_ansatz function exists
    if 'create_ansatz' not in namespace:
        return 0.0, "Error: create_ansatz function not defined"
    
    create_ansatz = namespace['create_ansatz']
    
    # Create ansatz
    try:
        qc_ansatz = create_ansatz(N)
    except Exception as e:
        return 0.0, f"Ansatz creation error: {str(e)}"
    
    # Validate ansatz is a QuantumCircuit
    if not isinstance(qc_ansatz, QuantumCircuit):
        return 0.0, f"Error: create_ansatz must return a QuantumCircuit, got {type(qc_ansatz)}"
    
    # Check parameter count
    L = len(qc_ansatz.parameters)
    if L > 9:
        return 0.0, f"Too many parameters: {L} > 9"
    
    # Generate Hamiltonian based on model_type
    if model_type == 'TF':
        H = generate_TF_hamiltonian(N, J, h, PBC)
    else:  # model_type == 'XY'
        H = generate_XY_hamiltonian(N, J, h, PBC)
    
    # Setup VQE
    optimizer = COBYLA(maxiter=3000, rhobeg=300, tol=1e-15)
    estimator = Estimator()
    
    # Suppress output during VQE
    f_stdout = io.StringIO()
    f_stderr = io.StringIO()
    
    # Evaluate with selected Hamiltonian
    try:
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            vqe = VQE(estimator=estimator, ansatz=qc_ansatz, optimizer=optimizer)
            result = vqe.compute_minimum_eigenvalue(operator=H)
            energy = result.eigenvalue
    except Exception as e:
        return 0.0, f"VQE computation error: {str(e)}"
    
    # Calculate score
    raw_score = -10 * energy
    
    # Calculate code length penalty
    code_length = len(idea_content)
    length_penalty = min(2 * (code_length // 40) / 10, 2)
    
    final_score = raw_score - length_penalty
    
    # Apply score floor: score cannot be negative
    final_score = max(final_score, 0.0)
    
    # Prepare info string
    info = (f"Model: {model_type}, Energy: {energy:.4f}, "
            f"Raw_score: {raw_score:.2f}, Length_penalty: {length_penalty:.2f}, "
            f"Final_score: {final_score:.2f}, "
            f"Params: {L}, Code_length: {code_length}")
    
    return round(final_score, 1), info


def evaluate(idea_content: str, model_type: str = None, timeout: float = 240.0):
    """
    Evaluate a quantum circuit ansatz idea using TF or XY Hamiltonian with timeout protection.
    
    Args:
        idea_content: String containing the code for create_ansatz function
        model_type: 'TF' for Transverse Field Ising, 'XY' for XY model. 
                    If None, uses global MODEL_TYPE variable.
        timeout: Timeout in seconds (default 240)
        
    Returns:
        tuple: (score: float, info: str or None)
            - score: VQE energy result (higher is better, multiplied by -10)
            - info: Additional information about the evaluation
    """
    
    # Use provided model_type or fall back to global variable
    if model_type is None:
        model_type = MODEL_TYPE
    
    # Validate model_type
    if model_type not in ['TF', 'XY']:
        return 0.0, f"Invalid model_type: {model_type}. Must be 'TF' or 'XY'"
    
    # Use threading to implement timeout
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = _evaluate_with_timeout(idea_content, model_type, timeout)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return 0.0, f"Evaluation timeout (>{timeout}s)"
    
    if exception[0] is not None:
        error_msg = str(exception[0])
        if len(error_msg) > 150:
            error_msg = error_msg[:150] + "..."
        return 0.0, f"Unexpected error: {error_msg}"
    
    return result[0] if result[0] is not None else (0.0, "Unknown error")


# For backward compatibility and testing
if __name__ == "__main__":
    try:
        # Test code
        test_code = """
```python
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_ansatz(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('params', 3)
    a, b, c = params[0], params[1], params[2]
    
    for qubit in range(n):
        qc.ry(a, qubit)
        qc.rz(b, qubit)
        qc.rx(c, qubit)
    
    for qubit in range(n - 1):
        qc.cz(qubit, qubit + 1)
    
    return qc
```
"""
        
        print("Testing TF model:")
        score_tf, info_tf = evaluate(test_code, model_type='TF')
        print(f"Score: {score_tf}")
        print(f"Info: {info_tf}")
        
        print("\nTesting XY model:")
        score_xy, info_xy = evaluate(test_code, model_type='XY')
        print(f"Score: {score_xy}")
        print(f"Info: {info_xy}")
    except ImportError as e:
        print(f"Qiskit not installed. This is expected in non-quantum environments.")
        print(f"The evaluation function will work when called from IdeaSearch with qiskit available.")
