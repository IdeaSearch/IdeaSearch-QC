import itertools
import warnings
import numpy as np
import re
import pathlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import scipy
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

# Suppress Qiskit's DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Code Extraction Utility ---
class CodeExtraction:
    """Extracts a Python function from a string, handling markdown code blocks."""
    def __init__(self, target_function_name="create_ansatz"):
        self.target_function_name = target_function_name
        self.import_code = self._load_dependency_code()

    def _load_dependency_code(self) -> str:
        """Loads required helper functions from a local file."""
        try:
            gates_file_path = pathlib.Path(__file__).parent / 'gates_function.py'
            with open(gates_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print("Warning: Dependency file 'gates_function.py' not found.")
            return ""

    def extract_function(self, code_string: str):
        """Parses a string, executes it, and returns the target function."""
        code_match = re.search(r"```python(.*?)```", code_string, re.DOTALL)
        code = code_match.group(1).strip() if code_match else code_string
        
        try:
            namespace = {}
            exec(self.import_code + '\n' + code, namespace)
            return namespace.get(self.target_function_name)
        except Exception as e:
            print(f"Failed to extract function from code: {e}")
            return None
        
    def extract_code_text(self, code_string: str):
        """
        Parses a string to find the Python code block, returning it as a string.
        It first looks for a ```python ... ``` blcokock, falling back to parsing the whole string.
        """
        code = code_string
        if match := re.search(r"```python(.*?)```", code_string, re.DOTALL):
            code = match.group(1).strip()
        return code

code_extraction = CodeExtraction()
extract_code_text = code_extraction.extract_code_text

# --- Core Logic ---

@lru_cache(maxsize=4) # Cache results to avoid re-computing for same parameters
def generate_hamiltonian(lamb: float = 0.0, nsite_root: int = 3, n_qubit: int = 1, phi_max=0.45) -> SparsePauliOp:
    """
    Generates the problem Hamiltonian with adjustable system size.
    Note here use phi max as 0.45, suitable for maximum entangled ground state.
    """
    # print(f"Generating Hamiltonian for nsite_root={nsite_root}, lambda={lamb}...")
    # --- System Parameters ---
    mass = 0.0
    nsite = nsite_root ** 2
    delta_y = 1.0
    delta_x = 1.0
    num_qubits = nsite * n_qubit

    # --- Derived Parameters ---
    delta_p = 2 * np.pi / (nsite_root * delta_x)
    n0 = (nsite_root - 1) // 2
    pmax = n0 * delta_p

    nphi = 2**n_qubit

    # Calculate momentum eigenvalues using numpy for clarity and performance
    pevc=list(range(nsite_root))
    for i in range(nsite_root):
        pevc[i] = -pmax + (i)*delta_p

    # Vectorized calculation of omega using a meshgrid
    px, py = np.meshgrid(pevc, pevc, indexing='ij')
    sin_sq_term = np.sin(px * delta_x / 2)**2 + np.sin(py * delta_y / 2)**2
    omega = np.sqrt(mass**2 + 4 * sin_sq_term / delta_x)

    # Calculate average omega
    omegabar = np.mean(omega)

    # Calculate phi_max
    phi_max_term = np.pi * (nphi - 1)**2 / (2 * nphi * delta_x * omegabar)
    if phi_max is None:
        phi_max = np.sqrt(phi_max_term)
        
    # print("phi_max =", phi_max)

    delta_phi = 2 * phi_max / (nphi - 1) if nphi > 1 else 2 * phi_max

    phi_i = [
        SparsePauliOp.from_sparse_list(
            [('Z', [i * n_qubit + j], -(delta_phi / 2) * (2**j)) for j in range(n_qubit)],
            num_qubits=num_qubits
        ) for i in range(nsite)
    ]
    phi_i.reverse() # Original code built this in reverse order

    # --- H_phi term (Spatial derivative) ---
    hphi = SparsePauliOp.from_sparse_list([], num_qubits=num_qubits)
    for i in range(nsite):
        # X-direction derivative (periodic boundary)
        i_minus_1 = (i - 1) if i % nsite_root != 0 else i + nsite_root - 1
        i_plus_1 = (i + 1) if i % nsite_root != nsite_root - 1 else i - nsite_root + 1
        hphi += (delta_x * delta_y / (2 * delta_x**2)) * (phi_i[i] @ (2 * phi_i[i] - phi_i[i_minus_1] - phi_i[i_plus_1]))

        # Y-direction derivative (periodic boundary)
        i_minus_nsr = (i - nsite_root) if i >= nsite_root else i + nsite - nsite_root
        i_plus_nsr = (i + nsite_root) if i < nsite - nsite_root else i - nsite + nsite_root
        hphi += (delta_x * delta_y / (2 * delta_y**2)) * (phi_i[i] @ (2 * phi_i[i] - phi_i[i_minus_nsr] - phi_i[i_plus_nsr]))

    # --- H_k term (Kinetic) ---
    from scipy.sparse import diags
    mat_size = 2**n_qubit

    # In our case, pi_square_mat is just 2*I+X
    pi_square_mat = diags([2], [0], shape=(mat_size, mat_size)).toarray()
    if mat_size > 1:
        pi_square_mat += diags([-1, -1], [-1, 1], shape=(mat_size, mat_size)).toarray()
        pi_square_mat[0, -1] = 1 # Anti-Periodic boundary
        pi_square_mat[-1, 0] = 1 # Anti-Periodic boundary

    pauli_matrices = {
        'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]])
    }

    pi_square_i_ops = [None] * nsite
    for i in range(nsite):
        lis_pi = []
        for pauli_str in [''.join(p) for p in itertools.product(pauli_matrices.keys(), repeat=n_qubit)]:
            full_pauli_op = pauli_matrices[pauli_str[0]]
            for char in pauli_str[1:]:
                full_pauli_op = np.kron(full_pauli_op, pauli_matrices[char])
            
            coeff = np.real(np.trace(pi_square_mat @ full_pauli_op)) / mat_size
            if abs(coeff) > 1e-9:
                lis_pi.append([pauli_str, list(range(i*n_qubit, (i+1)*n_qubit)), coeff])
        pi_square_i_ops[nsite-1-i] = SparsePauliOp.from_sparse_list(lis_pi, num_qubits=num_qubits)

    hk = SparsePauliOp.from_sparse_list([], num_qubits=num_qubits)
    for i in range(nsite):
        hk += (delta_x * delta_y / (2 * delta_phi**2)) * pi_square_i_ops[i]

    # --- H_int term (Interaction) ---
    h_int = SparsePauliOp.from_sparse_list([], num_qubits=num_qubits)
    if abs(lamb) > 1e-9:
        for i in range(nsite):
            h_int += lamb * (delta_x * delta_y) * (phi_i[i] @ phi_i[i] @ phi_i[i])

    return hk + hphi + h_int

def _get_ground_state(lamb: float = 0.0, nsite_root: int = 3, n_qubit: int = 1, phi_max=0.45, cache_dir: pathlib.Path = None, use_cache: bool = True) -> tuple[Statevector, float]:
    """
    Numerically calculates the ground state and its energy, with on-disk caching.
    Tries to load the result from a .npz file first. If not found, it computes
    the ground state and saves the result for future use.

    Args:
        lamb (float): The lambda parameter for the Hamiltonian.
        nsite_root (int): The square root of the number of sites.
        n_qubit (int): The number of qubits per site.
        cache_dir (pathlib.Path, optional): The directory to store cache files.
        use_cache (bool): If True, enables loading from and saving to cache.
    
    Returns:
        tuple[Statevector, float]: A tuple containing the ground state vector
                                   and its corresponding energy.
    """
    if use_cache:
        if cache_dir is None:
            cache_dir = pathlib.Path(__file__).parent / "ground_states"
        cache_dir.mkdir(exist_ok=True)
        # Use a consistent file naming convention, including n_qubit
        filename = f"ground_state_nsr{nsite_root}_lamb{lamb:.2f}_phimax{phi_max:.3f}.npz"
        cache_file = cache_dir / filename

        # Try to load from cache first
        if cache_file.exists():
            print(f"Loading ground state from cache: {cache_file}")
            try:
                data = np.load(cache_file)
                ground_state_vector = data['vector']
                ground_state_energy = data['energy']
                print(f"   Cached Ground State Energy: {ground_state_energy:.6f}")
                return Statevector(ground_state_vector), ground_state_energy
            except Exception as e:
                print(f"Warning: Could not load cache file {cache_file}. Recalculating. Error: {e}")

    # If not in cache or caching is disabled, perform the calculation
    print(f"Calculating ground state (caching {'enabled' if use_cache else 'disabled'}) for λ={lamb}, nsite_root={nsite_root}, n_qubit={n_qubit}...")
    H_op = generate_hamiltonian(lamb=lamb, nsite_root=nsite_root, n_qubit=n_qubit, phi_max=phi_max)
    H_matrix = H_op.to_matrix(sparse=True)
    
    # Use SciPy to find the lowest eigenvalue and eigenvector
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H_matrix, k=1, which='SA')
    ground_state_energy = eigenvalues[0]
    ground_state_vector = eigenvectors[:, 0]
    
    # Save the result to cache if caching is enabled
    if use_cache:
        try:
            print(f"   Saving ground state to cache: {cache_file}")
            np.savez(cache_file, vector=ground_state_vector, energy=ground_state_energy)
        except Exception as e:
            print(f"Warning: Could not save cache file {cache_file}. Error: {e}")

    print(f"   True Ground State Energy: {ground_state_energy:.6f}")
    return Statevector(ground_state_vector), ground_state_energy

# test module
if __name__ == "__main__":
  for nsite_root in [2, 3, 4, 5]:
      _, ground_energy = _get_ground_state(0.2, nsite_root)
      print(f"Ground energy for nsite_root={nsite_root}: {ground_energy}")