from qiskit import QuantumCircuit
from typing import Callable, List

def single_qubit_gates(
    circ: QuantumCircuit,
    gate_function: Callable[[QuantumCircuit, int], None]
):
    """
    Applies a specified single-qubit gate function sequentially to each qubit in the circuit.

    Args:
        circ (QuantumCircuit): The quantum circuit to operate on.
        gate_function (Callable[[QuantumCircuit, int], None]): A function that
            takes a QuantumCircuit and a qubit index as input, and adds a gate to that qubit.
    """
    num_qubits = circ.num_qubits
    for i in range(num_qubits):
        gate_function(circ, i)

from qiskit.circuit import Parameter
from typing import Union
def c4_symmetric_rzz(
    circ: QuantumCircuit,
    theta: Union[float, Parameter],
    nsite_root: int = 3
):
    """
    Applies an RZZ gate to a lattice with periodic boundary conditions according to C4 rotational symmetry.
    This includes applying the same angle theta to all horizontal and vertical nearest-neighbor couplings.

    Args:
        circ (QuantumCircuit): The quantum circuit to operate on.
        theta (Union[float, Parameter]): The rotation angle for the RZZ gate.
        nsite_root (int): The side length of the lattice (e.g., 3 for a 3x3 lattice).
    """
    total_qubits = nsite_root * nsite_root
    if circ.num_qubits != total_qubits:
        raise ValueError("The total number of qubits in the circuit does not match the lattice size.")

    # 1. Horizontal direction coupling (including periodic boundaries)
    for row in range(nsite_root):
        for col in range(nsite_root):
            q1 = row * nsite_root + col
            # Periodic boundary condition: (col + 1) % nsite_root
            q2 = row * nsite_root + (col + 1) % nsite_root
            circ.crz(theta, q1, q2)

    # 2. Vertical direction coupling (including periodic boundaries)
    for col in range(nsite_root):
        for row in range(nsite_root):
            q1 = row * nsite_root + col
            # Periodic boundary condition: (row + 1) % nsite_root
            q2 = ((row + 1) % nsite_root) * nsite_root + col
            circ.crz(theta, q1, q2)
