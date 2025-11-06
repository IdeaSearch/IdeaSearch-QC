####################################
# circuit_design/run.py                        
# Updated on 2025/07/17              
####################################
from IdeaSearch import IdeaSearcher
from evaluation import evaluate
from prompt import system_prompt, prologue_section, epilogue_section, filter

# replace the following path with your own paths.
api_path = "Your api key position" #"/data/sonny/ideas/api_keys.json"
data_path = "Your database path" #"/data/sonny/ideas/QC_lam0p2/optimize/database"
models = [
        "gpt-4o",
        "gpt-4o",
        "gemini-flash",
        "gemini-pro",
        "gemini-flash",
        "Deepseek_V3",
    ]
model_temperatures = [
        0.9,
        0.3,
        0.4,
        0.9,
        0.9,
        0.8,
    ]

def main():
    
    ideasearcher = IdeaSearcher()
    
    # load models
    ideasearcher.set_api_keys_path(api_path)
    
    # set minimum required parameters
    ideasearcher.set_program_name("QC with lambda=0.2")
    ideasearcher.set_database_path(data_path)
    ideasearcher.set_evaluate_func(evaluate)
    ideasearcher.set_prologue_section('You will next generate a creative new function based on the prompt, generating a function that is as good as possible according to the function format given in the prompt')
    ideasearcher.set_epilogue_section('Now start generating')
    ideasearcher.set_filter_func(filter)
    ideasearcher.set_sample_temperature(100.0)
    ideasearcher.set_examples_num(2)
    ideasearcher.set_models(models)
    ideasearcher.set_model_temperatures(model_temperatures)
    ideasearcher.set_model_sample_temperature(20.0)
    ideasearcher.set_model_assess_window_size(5)
    ideasearcher.set_hand_over_threshold(0.1)
    ideasearcher.set_model_assess_average_order(15.0)
    ideasearcher.add_initial_ideas([
        """
I will design a novel variational quantum ansatz for you, named **"Quantum Relaxation Lattice Field Ansatz (QRLFA)"**. This design uses phased quantum operations and clever parameter sharing to efficiently simulate the ground state of a scalar boson field on a 3x3 lattice, while strictly adhering to hardware efficiency and symmetry principles.

---

### Novel Ansatz Design: **Quantum Relaxation Lattice Field Ansatz (QRLFA)**

#### Physical Motivation and Core Concept

This design is based on the physical intuition of "phased relaxation":
1.  **Initial Field Excitation and Quantum Fluctuations:** The ground state of field theory is not simply a zero field, but is filled with quantum fluctuations. We first excite the system from the zero state to a superposition state with universal quantum fluctuations through single-qubit gates, simulating the initial "imaginary time evolution" or "quenching" process.
2.  **Core Kinetic Coupling:** After establishing initial field excitation, introduce a core nearest-neighbor entanglement layer to precisely simulate the kinetic term of the field, allowing interactions between lattice sites to be established.
3.  **Energy Relaxation and Symmetry Convergence:** Finally, fine-tune the system through another set of single-qubit gates. This is not just simple parameter correction, but a "quantum relaxation" mechanism aimed at guiding the system to the lowest energy ground state, and promoting its convergence to the symmetric state of `⟨ϕ⟩=0` through nonlinear correlations between parameters. This phased operation simulates the evolution of a physical system from a high-energy random state to a low-energy ground state.

=====================================================================

#### Code Implementation

```python
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from typing import Callable, Union

# Assume single_qubit_gates and c4_symmetric_rzz are predefined in the execution environment
# You do not need to and are prohibited from importing or redefining them in your code.

def create_ansatz(nsite_root=3):

    total_qubits = nsite_root ** 2
    circ = QuantumCircuit(total_qubits)
    
    # Use only 3 shared parameters to achieve parameter economy
    params = ParameterVector('theta', 3)

    # 1. Initial Field Preparation Layer
    # Physical motivation: Rotate each qubit from |0> to a quantum superposition state through RY gates.
    # This simulates the initial quantum fluctuations of the field without interactions or the starting point of "imaginary time evolution".
    # theta[0] controls the "amplitude" or "strength" of the initial field.
    single_qubit_gates(circ, lambda c, i: c.ry(params[0], i))
    
    # 2. Local Phase Tuning Layer
    # Physical motivation: Adjust the local phase of each qubit through RZ gates.
    # This provides a more refined starting point for subsequent RZZ entanglement, and can also be seen as an initial phase contribution to the mass term.
    # theta[1] introduces independent phase control.
    single_qubit_gates(circ, lambda c, i: c.rz(params[1], i))
    
    # 3. Core Kinetic Coupling Layer
    # Physical motivation: Use c4_symmetric_rzz gates to establish entanglement between all horizontal and vertical nearest neighbors.
    # This is the key operation for simulating the (∇ϕ)² kinetic term in the Hamiltonian, ensuring spatial correlations.
    # theta[2] directly controls the strength of kinetic coupling.
    c4_symmetric_rzz(circ, params[2], nsite_root)
    
    # 4. Quantum Relaxation & Symmetry Restoration Layer
    # Physical motivation: This layer aims to guide the system to "relax" toward the lowest energy ground state and help restore the symmetry of ⟨ϕ⟩=0.

    # RY(-theta[0]): This is a key "reverse" or "correction" rotation.
    # It aims to guide the system to converge to the symmetric ground state of ⟨ϕ⟩=0 by counteracting or adjusting the influence introduced by the initial RY(theta[0]).
    # This "explore first, then converge" pattern helps find an unbiased ground state.
    single_qubit_gates(circ, lambda c, i: c.ry(-params[0], i))
    
    # RZ(theta[0] + theta[1]): Combine the first two parameters for final phase fine-tuning.
    # This nonlinear combination (parameter addition) increases expressiveness, allowing the ansatz to
    # explore more complex wavefunction phase spaces without increasing the number of parameters,
    # thereby more accurately approximating the ground state energy.
    single_qubit_gates(circ, lambda c, i: c.rz(params[0] + params[1], i))
    
    return circ
```

#### Design Element Analysis

1.  **How to Simulate Physical Terms (Mass Term and Kinetic Term)**
    *   **Kinetic Term `(∇ϕ)²`:** This term describes the interaction between adjacent lattice site fields, with translational and C4 rotational symmetry. My design precisely and efficiently simulates it through **a single call to `c4_symmetric_rzz(circ, params[2])`**. `params[2]` directly controls the strength of all nearest-neighbor couplings, perfectly matching the physical model's description of the kinetic term.
    *   **Mass Term `m²ϕ²`:** This term represents the local potential energy at each lattice site, with translational symmetry. I use the `single_qubit_gates` function to apply `RY` and `RZ` rotations to simulate it.
        *   `RY(theta[0])` and `RY(-theta[0])` layers: Directly adjust the amplitude distribution of each qubit, which directly affects the expectation value `⟨ϕ⟩` and squared expectation value `⟨ϕ²⟩` of the field `ϕ`. Through initial excitation and subsequent "reverse relaxation", we can shape `⟨ϕ⟩` to approach zero while maintaining symmetry, and adjust `⟨ϕ²⟩` to match the effect of the mass term.
        *   `RZ(theta[1])` and `RZ(theta[0] + theta[1])` layers: Fine-tune the phase of the state. In quantum mechanics, the energy of the ground state depends not only on amplitude but also on phase distribution. These layers are crucial for finding the lowest energy point, especially since `RZ` gates can be seen as "energy terms" acting in the `Z` basis.

2.  **How Parameter Sharing Strategy Reduces Optimization Difficulty**
    My design uses only **3 parameters** (`theta[0]`, `theta[1]`, `theta[2]`), achieving extremely high parameter efficiency:
    *   `theta[2]` is specifically used to control the strength of the **kinetic term**, with clear and independent physical meaning.
    *   `theta[0]` and `theta[1]` are cleverly used for **mass term** adjustment and introduce complex nonlinear relationships:
        *   `theta[0]` not only controls the initial `RY` gate, but is also used for subsequent `RY(-theta[0])` (reverse operation) and `RZ(theta[0] + theta[1])` (nonlinear combination). This reuse forces the optimizer to find a `theta[0]` value that can simultaneously satisfy multiple physical requirements (initial excitation, relaxation, final phase adjustment), thereby enhancing the ansatz's expressiveness with very few parameters.
        *   `theta[1]` controls the initial `RZ` gate and is combined with `theta[0]` for the final `RZ` gate. This allows two independent single-qubit gate parameters to interact in the final layer, creating richer quantum states.
    This parameter sharing strategy effectively reduces the number of independent variational parameters by establishing complex functional dependencies between different layers and different gate types, thereby greatly reducing the difficulty and convergence time of VQE optimization while avoiding overfitting.

3.  **Why Your Layer Structure Design is Effective**
    My QRLFA ansatz adopts a phased "local-global-local" operation pattern, aimed at efficiently and purposefully constructing the target ground state:
    *   **Phase 1 (Initial Excitation and Local Preparation):** `RY(theta[0])` and `RZ(theta[1])` layers.
        *   Starting from the `|0...0>` state, rotate each qubit to a superposition state through `RY` gates, simulating the quantum fluctuations universally present in field theory, providing a starting point closer to the physical true ground state for subsequent entanglement, rather than a simple classical zero field. The `RZ` gate further adjusts the local phase, preparing for the core entanglement layer. This ensures the ansatz can start exploring Hilbert space from a good, symmetric initial point.
    *   **Phase 2 (Core Kinetic Coupling):** `c4_symmetric_rzz(circ, params[2])` layer.
        *   This is the core of the ansatz, efficiently and symmetrically establishing nearest-neighbor correlations between lattice sites through the unique `c4_symmetric_rzz` operation, directly simulating the kinetic term of the Hamiltonian. Since its hardware overhead is large, using it only once ensures the circuit depth and number of CX gates remain within acceptable ranges.
    *   **Phase 3 (Quantum Relaxation and Fine-tuning):** `RY(-theta[0])` and `RZ(theta[0] + theta[1])` layers.
        *   This phase is the innovative point of the design. After the system has established spatial correlations, instead of blindly increasing entanglement, fine local single-qubit operations are used to "relax" the system:
            *   `RY(-theta[0])`: Simulates the process of the system "self-adjusting" to reach the lowest energy point in field theory. It attempts to correct or counteract the deviation brought by the initial `RY(theta[0])`, guiding the system to converge to the symmetric ground state of `⟨ϕ⟩=0`.
            *   `RZ(theta[0] + theta[1])`: Through nonlinear combination of parameters, provides more complex phase modulation capability without increasing additional parameters, thereby precisely adjusting the wavefunction to better match the fine structure of the ground state, suppressing higher energy components through interference effects.

This layering and parameter strategy enables QRLFA to achieve powerful expressiveness and optimization convergence with minimal parameters while strictly adhering to symmetry and hardware efficiency, and is expected to achieve excellent performance in solving the 3x3 scalar field ground state problem.
"""
    ])

    # add 10 islands
    for _ in range(4):
        ideasearcher.add_island()
    
    
    # Evolve for 11 cycles, 25 epochs on each island per cycle with ideas repopulated at the end
    for cycle in range(4):
        if cycle % 2 == 0:
            ideasearcher.set_filter_func(None)
        else:
            ideasearcher.set_filter_func(filter)
        if cycle % 5 == 0:
            ideasearcher.set_system_prompt(system_prompt)
            ideasearcher.set_prologue_section(prologue_section)
            ideasearcher.set_epilogue_section(epilogue_section)
        
        ideasearcher.run(25)
        ideasearcher.repopulate_islands()
        
        best_idea = ideasearcher.get_best_idea()
        best_score = ideasearcher.get_best_score()
        print(
            f"[Round {cycle+1}] "
            f"Current highest score: {best_score:.2f}, this idea is:\n"
            f"{best_idea}\n"
        )
        
        if cycle % 5 == 4:
            ideasearcher.set_system_prompt('')
            ideasearcher.set_prologue_section('You will next design a creative new function based on the prompt function, partially inheriting previous characteristics, but must not generate content that is exactly the same')
            ideasearcher.set_epilogue_section('Start generating')


if __name__ == "__main__":
    main()
