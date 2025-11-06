# QC_lam0p2/prompt.py

__all__ = [
    "system_prompt",
    "prologue_section",
    "epilogue_section",
    "filter"
]

from evaluation import extract_code_text
filter = extract_code_text

system_prompt = (
    "You are a top-tier quantum physicist and algorithm expert, specializing in designing efficient, symmetric quantum circuits for lattice quantum field theory. "
    "Your task is to design a novel, hardware-efficient ansatz for a 3x3 scalar boson field theory. You must use the provided high-level symmetry functions to construct the circuit, ensuring translational invariance and rotational invariance. "
    "Please go beyond existing examples and innovate."
)

prologue_section = (
    "### Research Background\n"
    "In quantum simulation and lattice field theory research, ground state preparation of two-dimensional bosonic fields is a key verification scenario for quantum advantage.\n"
    "Existing methods face the following challenges:\n"
    "- Traditional variational circuit parameters grow rapidly with system size\n"
    "- Field theory symmetries are difficult to maintain in quantum circuits\n"
    "- Convergence in the continuum limit is difficult to guarantee\n\n"
    
    "### Physical System Description\n"
    "We study scalar field theory on a 3×3 lattice with the following properties:\n"
    "1. Each lattice site field value ϕ is encoded using 1 qubit (discretization scheme)\n"
    "2. The Hamiltonian includes nearest-neighbor coupling terms (∇ϕ)² and mass term m²ϕ²\n"
    "3. Periodic boundary conditions\n"
    "4. The target state should satisfy ⟨ϕ⟩=0 and correlation functions exhibit exponential decay\n"
    "5. The continuum limit of the target state is a Gaussian wave packet\n\n"

    "### Available High-Level Symmetry Functions (Symmetry-Aware Building Blocks)\n"
    "To ensure circuit symmetry and simplify design, you **must** use the following predefined high-level functions to construct your ansatz. **You do not need to and are prohibited from importing or redefining them** in your code; they are already available in the execution environment.\n\n"
    
    "#### 1. Single-Qubit Gate Layer (Enforces Translational Symmetry)\n"
    "```python\n"
    "def single_qubit_gates(\n"
    "    circ: QuantumCircuit,\n"
    "    gate_function: Callable[[QuantumCircuit, int], None]\n"
    "):\n"
    '    """\n'
    "    Applies a specified single-qubit gate function sequentially to each qubit in the circuit.\n"
    "    This enforces translational invariance of the lattice, corresponding to mass terms or external fields.\n"
    "    Usage example: \n"
    "    # Apply RX gate with the same angle to all qubits\n"
    "    single_qubit_gates(circ, lambda c, i: c.rx(params[0], i))\n"
    '    """\n'
    "```\n\n"

    "#### 2. Two-Qubit Gate Layer (Enforces C4 Rotation and Translational Symmetry)\n"
    "```python\n"
    "def c4_symmetric_rzz(\n"
    "    circ: QuantumCircuit,\n"
    "    theta: Union[float, Parameter],\n"
    "    nsite_root: int = 3\n"
    "):\n"
    '    """\n'
    "    Applies an RZZ gate to all horizontal and vertical nearest neighbors according to C4 rotational symmetry and translational invariance.\n"
    "    This automatically handles periodic boundary conditions and is ideal for constructing the (∇ϕ)² kinetic term.\n"
    "    Usage example:\n"
    "    # Apply RZZ interaction between all nearest neighbors with the same angle\n"
    "    c4_symmetric_rzz(circ, params[1])\n"
    '    """\n'
    "```\n\n"

    "### Reference Examples\n"
    "The following are verified effective ansatz structure examples (they are also constructed using similar high-level functions):\n"
)

# --- epilogue_section with reinforced rules ---
epilogue_section = (
    "### Output Requirements\n"
    "1. Propose a new ansatz structure and explain its physical motivation.\n"
    "2. Implement the final circuit in Python code, defined using create_ansatz(nsite_root=3), without introducing any additional parameters here.\n"
    "3. Explain:\n"
    "   - How the parameter sharing strategy reduces optimization difficulty.\n"
    "   - Why your layer structure design is effective.\n\n"
    
    "### **Core Design Rules**\n"
    "1.  **Primary Rule: You must and can only use the two high-level functions `single_qubit_gates` and `c4_symmetric_rzz` to construct your ansatz layers.** Do not manually add low-level gates like `circ.rx`, `circ.crz`, etc.\n"
    "2.  **Parameter Sharing Strategy**: Share parameters as much as possible to reduce optimization difficulty. For example, different calls to `single_qubit_gates` may use the same parameters to reduce the parameter count.\n"
    "3.  **Circuit Evolution**: The circuit starts from the `|00...0>` state. A potentially good strategy is to first use `X` and `H` gates to transform all qubits to the `|-... -->` state (implemented using `single_qubit_gates`), then begin applying variational layers.\n"
    "4.  **Innovation**: **Do not** copy reference examples, **do not** generate identical circuits. Try different layer combination orders, different single-qubit gates, or design a novel multi-layer structure.\n\n"
    "5.  **Cost Considerations**: For nsite_root=3, the cost of using `single_qubit_gates` is approximately 5% of `c4_symmetric_rzz`, however `c4_symmetric_rzz` is the only way to generate entanglement. A wise approach is to design single-qubit gates as complexly as possible."
    "Now start designing your quantum ansatz!"
)
