__all__ = [
    "evaluate",
    "extract_code_text"
]

import warnings
import numpy as np
import re
import pathlib
import time
import multiprocessing
import queue
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

# Suppress Qiskit's DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hamiltonian
import sys
root_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_path))

from hamiltonian import generate_hamiltonian, _get_ground_state, CodeExtraction

# --- Code Extraction Utility ---

def extract_code_text(code_string: str):
    """
    Parses a string to find the Python code block, returning it as a string.
    It first looks for a ```python ... ``` blcokock, falling back to parsing the whole string.
    """
    code = code_string
    if match := re.search(r"```python(.*?)```", code_string, re.DOTALL):
        code = match.group(1).strip()
    return code

# --- Core Logic ---

def run_vqe_instance(params):
    """Helper function for running a single VQE instance in a separate process."""
    initial_point, qc_ansatz, H = params
    estimator = Estimator()
    optimizer = COBYLA(maxiter=1000)
    vqe = VQE(estimator=estimator, ansatz=qc_ansatz, optimizer=optimizer, initial_point=initial_point)
    return vqe.compute_minimum_eigenvalue(operator=H)

class QuantumCircuitEvaluator:
    """
    A comprehensive evaluator for quantum circuit ideas using VQE.
    
    This class encapsulates the entire workflow:
    1.  Sets up the environment and problem Hamiltonian.
    2.  Builds and optimizes an ansatz circuit from an idea string.
    3.  Runs multiple VQE simulations to find the minimum energy.
    4.  Analyzes the results for fidelity, depth, and gate counts.
    5.  Generates detailed reports with scores, diagnostics, and visualizations.
    """
    def __init__(self, idea: str, lamb: float, nsite_root: int, phi_max: float, num_vqe_runs: int, save_report: bool = True):
        self.idea = idea
        self.lamb = lamb
        self.nsite_root = nsite_root
        self.phi_max = phi_max
        self.num_vqe_runs = num_vqe_runs
        self.save_report = save_report
        
        self.output_dir = pathlib.Path(__file__).parent / f"eval_results_{time.strftime('%Y%m%d-%H%M%S')}"
        self.metrics = {}
        self.code_extractor = CodeExtraction()

    def _prepare_environment(self):
        """Creates the output directory for reports and visualizations if saving is enabled."""
        if self.save_report:
            self.output_dir.mkdir(exist_ok=True)

    def _build_and_optimize_ansatz(self):
        """Extracts, builds, and optimizes the ansatz circuit."""
        create_ansatz_fn = self.code_extractor.extract_function(self.idea)
        if not callable(create_ansatz_fn):
            raise ValueError("Failed to extract 'create_ansatz' function from the idea.")
        
        self.qc_initial = create_ansatz_fn(self.nsite_root)
        if self.qc_initial.num_parameters == 0:
            raise ValueError("Ansatz has zero parameters.")
        
        self.metrics['initial_depth'] = self.qc_initial.depth()
        self.metrics['initial_ops'] = self.qc_initial.count_ops()
        self._plot_circuit(self.qc_initial, "initial_circuit")

        # Remove barriers before optimization (compatible with older Qiskit versions)
        new_data = [instr for instr in self.qc_initial.data if instr.operation.name != 'barrier']
        self.qc_initial.data = new_data

        try:
            pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
            self.qc_optimized = pass_manager.run(self.qc_initial)
            print("After Pass Manager, depth is:", self.qc_optimized.depth())
        except Exception as e:
            print(f"Warning: PassManager optimization failed with error: {e}. Using initial circuit.")
            self.qc_optimized = self.qc_initial
            
        self.metrics['num_params'] = self.qc_optimized.num_parameters
        if self.metrics['num_params'] == 0:
            raise ValueError("Optimized ansatz has zero parameters.")
        
        return True

    def _run_vqe_simulations(self):
        """Runs VQE simulations in parallel."""
        self.H = generate_hamiltonian(lamb=self.lamb, nsite_root=self.nsite_root, phi_max=self.phi_max)
        self.ground_state, self.metrics['true_ground_energy'] = _get_ground_state(lamb=self.lamb, nsite_root=self.nsite_root, phi_max=self.phi_max)
        
        initial_points = [np.random.uniform(-np.pi, np.pi, self.metrics['num_params']) for _ in range(self.num_vqe_runs)]
        
        with ProcessPoolExecutor() as executor:
            tasks = [(p, self.qc_optimized, self.H) for p in initial_points]
            self.vqe_results = list(executor.map(run_vqe_instance, tasks))
        
        self.metrics['vqe_energies'] = [r.eigenvalue for r in self.vqe_results]
        self.best_result = min(self.vqe_results, key=lambda r: r.eigenvalue)
        self.metrics['vqe_energy'] = self.best_result.eigenvalue
        self.metrics['energy_std_dev'] = np.std(self.metrics['vqe_energies'])

    def _run_extrapolation_check(self):
        """
        If nsite_root is 3, run VQE for nsite_root=2 and 4 using the best
        parameters from the nsite_root=3 run as the initial point.
        This is to check the extrapolation performance of the ansatz.
        """
        if self.nsite_root != 3:
            self.metrics['extrapolation'] = {} # Initialize to avoid key errors
            return

        print("Running extrapolation check for nsite_root=2 and nsite_root=4...")
        
        if not hasattr(self.best_result, 'optimal_point') or self.best_result.optimal_point is None:
            print("Warning: Could not find optimal_point from VQE result. Skipping extrapolation check.")
            self.metrics['extrapolation'] = {}
            return
            
        initial_point = self.best_result.optimal_point
        print(f"Parameters of VQE for 3x3 sites: {initial_point}")
        self.metrics['extrapolation'] = {}

        for nsite_root_ext in [2, 4]:
            try:
                # 1. Build ansatz
                create_ansatz_fn = self.code_extractor.extract_function(self.idea)
                qc_ansatz = create_ansatz_fn(nsite_root_ext)
                
                # Optimize ansatz
                pass_manager = PassManager([Optimize1qGates(), CommutativeCancellation()])
                qc_ansatz = pass_manager.run(qc_ansatz)

                if qc_ansatz.num_parameters != len(initial_point):
                    print(f"Warning: Parameter number mismatch for nsite_root={nsite_root_ext}. "
                          f"Expected {qc_ansatz.num_parameters}, got {len(initial_point)}. "
                          "Skipping extrapolation check for this size.")
                    self.metrics['extrapolation'][f'fidelity_{nsite_root_ext}'] = 0.0
                    continue

                # 2. Get Hamiltonian and ground state
                H_ext = generate_hamiltonian(lamb=self.lamb, nsite_root=nsite_root_ext, phi_max=self.phi_max)
                ground_state_ext, _ = _get_ground_state(lamb=self.lamb, nsite_root=nsite_root_ext, phi_max=self.phi_max)

                # 3. Run VQE (once)
                vqe_result_ext = run_vqe_instance((initial_point, qc_ansatz, H_ext))
                print(f"Parameters of VQE for nsite_root={nsite_root_ext}: {vqe_result_ext.optimal_point}")

                # 4. Calculate fidelity
                final_circuit_ext = qc_ansatz.assign_parameters(vqe_result_ext.optimal_parameters)
                final_state_ext = Statevector.from_instruction(final_circuit_ext)
                fidelity_ext = np.real(np.abs(final_state_ext.inner(ground_state_ext))**2)

                self.metrics['extrapolation'][f'fidelity_{nsite_root_ext}'] = fidelity_ext
                print(f"Extrapolation for nsite_root={nsite_root_ext} complete. Fidelity: {fidelity_ext:.4f}")

            except Exception as e:
                print(f"Error during extrapolation check for nsite_root={nsite_root_ext}: {e}")
                self.metrics['extrapolation'][f'fidelity_{nsite_root_ext}'] = 0.0

    def _analyze_and_transpile(self):
        """Analyzes the best result and transpiles the final circuit."""
        # Fidelity
        self.final_circuit = self.qc_optimized.assign_parameters(self.best_result.optimal_parameters)
        final_state = Statevector.from_instruction(self.final_circuit)
        self.metrics['fidelity'] = np.real(np.abs(final_state.inner(self.ground_state))**2)
        print("VQE fidelity for nsite_root=3:", self.metrics['fidelity'])

        # Transpilation
        basis_gates = ['cx', 'rx', 'ry', 'rz']
        self.qc_transpiled = transpile(self.final_circuit, basis_gates=basis_gates, optimization_level=2)
        self.metrics['transpiled_depth'] = self.qc_transpiled.depth()
        self.metrics['transpiled_ops'] = self.qc_transpiled.count_ops()
        self.metrics['transpiled_cx_gates'] = self.metrics['transpiled_ops'].get('cx', 0)
        self._plot_circuit(self.qc_transpiled, "transpiled_circuit")

    def _generate_report(self):
        """Generates all plots and the final text report."""
        self._plot_vqe_convergence()
        self._plot_parameter_distribution()
        
        score, report_text = self._calculate_score_and_create_report_text()
        self.metrics['score'] = score
        
        if self.save_report:
            report_path = self.output_dir / "evaluation_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
        return score, report_text

    def _plot_circuit(self, qc: QuantumCircuit, name: str):
        """Saves a circuit diagram if saving is enabled."""
        if not self.save_report:
            return
        try:
            path = self.output_dir / f"{name}.png"
            qc.draw('mpl', style='iqx', fold=-1).savefig(path, dpi=150)
        except Exception as e:
            print(f"Could not generate plot for {name}: {e}")

    def _plot_vqe_convergence(self):
        """Plots and saves the VQE energy convergence."""
        if not self.save_report:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['vqe_energies'], 'o-', label='VQE Energy per Run', alpha=0.7)
        plt.axhline(y=self.metrics['true_ground_energy'], color='r', linestyle='--', label='True Ground Energy')
        plt.axhline(y=self.metrics['vqe_energy'], color='g', linestyle='--', label='Best VQE Energy')
        plt.xlabel("VQE Run Instance")
        plt.ylabel("Energy")
        plt.title("VQE Energy Convergence")
        plt.legend()
        plt.grid(True)
        path = self.output_dir / "vqe_convergence.png"
        plt.savefig(path)
        plt.close()

    def _plot_parameter_distribution(self):
        """Plots and saves the distribution of optimal parameters."""
        if not self.save_report:
            return
        params = np.array(list(self.best_result.optimal_parameters.values()))
        normalized_params = np.mod(params, 2 * np.pi)
        
        plt.figure(figsize=(10, 6))
        plt.hist(normalized_params, bins=20, density=True, alpha=0.7, label='Normalized Parameters')
        plt.xlabel("Parameter Value (radians)")
        plt.ylabel("Density")
        plt.title("Distribution of Optimal Parameters (Normalized to [0, 2π])")
        plt.grid(True)
        plt.legend()
        path = self.output_dir / "parameter_distribution.png"
        plt.savefig(path)
        plt.close()

    def _calculate_score_and_create_report_text(self):
        """Calculates the final score and generates a detailed report string."""
        m = self.metrics
        m['energy_difference'] = m['vqe_energy'] - m['true_ground_energy']

        # --- Penalties ---
        penalties = {}
        penalties['energy'] = max(0.0, 10 * m["energy_difference"])
        penalties['fidelity'] = max(0.0, 80 * (0.95 - m["fidelity"])) ** 2
        penalties['depth'] = max(0.0, 0.8 * (m["transpiled_depth"] - 16))
        penalties['cx_gate'] = max(0.0, 0.3 * (m["transpiled_cx_gates"] - 40))
        penalties['param_num'] = (max(0.0, 1.5 * (m["num_params"] - 4)))**2
        
        stability_penalty = 0.0
        if m["energy_std_dev"] > 0.5: stability_penalty = 5.0
        elif m["energy_std_dev"] > 0.2: stability_penalty = 2.0
        penalties['stability'] = stability_penalty

        # --- Extrapolation Penalties ---
        extrapolation_section = ""
        if self.nsite_root == 3 and 'extrapolation' in m and m['extrapolation']:
            fid_2 = m['extrapolation'].get('fidelity_2', 0.0)
            fid_4 = m['extrapolation'].get('fidelity_4', 0.0)
            
            penalties['extrapolation_2'] = min(max(0.0, 50 * (0.99 - fid_2))**2, 45.0)
            penalties['extrapolation_4'] = min(max(0.0, 150 * (0.94 - fid_4))**2, 45.0)
            
            extrapolation_section = f"""
--- EXTRAPOLATION ANALYSIS (for nsite_root=3) ---
  2x2 Extrapolate Pen.:  - {penalties['extrapolation_2']:6.2f} (Fidelity at 2x2: {fid_2:.4f})
  4x4 Extrapolate Pen.:  - {penalties['extrapolation_4']:6.2f} (Fidelity at 4x4: {fid_4:.4f})
"""
        
        total_penalty = sum(penalties.values())
        score = max(0.0, 100 - total_penalty)

        # --- Diagnostics ---
        hints = []
        if penalties['stability'] > 0: hints.append(f"VQE Optimization Unstable (Energy StdDev={m['energy_std_dev']:.3f}). High variance suggests the optimization is unstable.")
        if penalties['energy'] > 15: hints.append(f"High Energy Gap (Penalty: {penalties['energy']:.1f}). VQE energy is far from the true ground state, indicating the ansatz may lack expressivity.")
        if penalties['fidelity'] > 15: hints.append(f"Low Fidelity (Penalty: {penalties['fidelity']:.1f}). The final state is not close to the true ground state. Your design should aim for >95% fidelity. Specifically, additional one qubit gates may be suitable for this problem.")
        if penalties['depth'] > 15: hints.append(f"High Transpiled Depth (Penalty: {penalties['depth']:.1f}). The circuit is very deep ({m['transpiled_depth']}), increasing noise on real hardware.")
        if penalties['cx_gate'] > 15: hints.append(f"Excessive CX Gates (Penalty: {penalties['cx_gate']:.1f}). The high number of CX gates ({m['transpiled_cx_gates']}) increases error rates.")
        if penalties['param_num'] > 15: hints.append(f"Too many parameters used (Penalty: {penalties['param_num']} ), you are ancouraged to find circuits with less than 7 parameters.")
        if penalties.get("extrapolation_2", 0.0) > 15: hints.append(f"High penalty on the 2x2 system ({penalties['extrapolation_2']:.2f}). A 2x2 lattice is a minimal benchmark; a circuit with sufficient expressibility is expected to achieve near-perfect performance on this scale.")
        if penalties.get("extrapolation_4", 0.0) > 15: hints.append(f"Poor generalization detected with a high extrapolation penalty at 4x4 size ({penalties['extrapolation_4']:.2f}). The goal is to design a robust circuit that maintains high fidelity across different system sizes.")

        ADVANCED_STRATEGY = r'''

=====================================================================
## Advanced Strategy:

**1. Context and Analysis of the Current Ansatz**

Based on analysis, a circuit architecture composed of a layer of single-qubit gates, an entangling layer, and a subsequent layer of single-qubit gates can be fully described by a 7-parameter ansatz. This is because any sequence of single-qubit gates can be composed into a single arbitrary SU(2) rotation, which requires three parameters. A concrete example of such a 7-parameter circuit is as follows:

```python
# Layer 1: Arbitrary SU(2) rotation (3 parameters)
single_qubit_gates(circ, lambda c, i: c.rx(params[0], i))
single_qubit_gates(circ, lambda c, i: c.ry(params[1], i))
single_ququbit_gates(circ, lambda c, i: c.rz(params[2], i))

# Layer 2: Entangling layer (1 parameter)
c4_symmetric_rzz(circ, params[3], nsite_root)

# Layer 3: Arbitrary SU(2) rotation (3 parameters)
single_qubit_gates(circ, lambda c, i: c.rx(params[4], i))
single_qubit_gates(circ, lambda c, i: c.ry(params[5], i))
single_qubit_gates(circ, lambda c, i: c.rz(params[6], i))
```

**2. Performance and Limitations**

However, the expressiveness of this ansatz does not guarantee its optimality. In a test conducted for a system with `nsite_root=3`, we utilized the Variational Quantum Eigensolver (VQE) to find the set of parameters that minimizes the energy and to prepare the corresponding variational state.

The results show that while the minimum energy achieved by this ansatz is closer to the true ground state energy, the prepared quantum state has a fidelity of only 90% with the actual ground state (defined as the absolute value of the inner product between the prepared state $|\psi_{prep}\rangle$ and the true ground state $|\psi_{true}\rangle$, i.e., $|\langle \psi_{true} | \psi_{prep} \rangle|$). This suggests that the variational state is a superposition of the ground state and low-lying excited states, with an insufficient population in the ground state component.

**3. Proposed Strategies**

**Strategy A: Ansatz Simplification**
A potentially effective strategy is to reduce the expressibility of the ansatz by using fewer parameters or a smaller number of single-qubit gates. The objective is to constrain the variational space, thereby limiting the inclusion of low-energy excited states in the final state. This may result in a higher energy expectation value (an "energy cost") but has the potential to increase the fidelity with the true ground state.

**Strategy B: Custom Circuit Design (High Risk)**
A second, more perilous, direction for improvement is to construct a circuit from fundamental gates. This approach should be considered only if there is strong reason to believe that the current ansatz is fundamentally incapable of representing the essential characteristics of the ground state.

This approach is **exceptionally risky** because a manually constructed circuit can easily fail to preserve the inherent symmetries of the ground state, namely the translational and C4 lattice symmetries. Therefore, you should only pursue a custom design after a thorough analysis indicates that this trade-off is necessary.

**4. Permissible Outputs**

In your design process, you may generate additional outputs. This can include requests for specific feedback on various aspects of the proposed circuit architecture.
'''
        if score < 52.0:
            hints.append(ADVANCED_STRATEGY)
        
        diagnostic_summary = "Key areas for improvement:\n" + "\n".join(f"- {h}" for h in hints) if hints else "Overall performance is good. Well done!"


        # --- Report String ---
        report_header = f"""
=====================================================================
                       EVALUATION REPORT
=====================================================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        if self.save_report:
            report_header += f"Output Directory: {self.output_dir}\n"

        report_body = f"""FINAL SCORE: {score:.2f}

--- SCORE BREAKDOWN (100 Base Score - Penalties) ---
  Energy Penalty:        - {penalties['energy']:6.2f} (Energy Gap: {m["energy_difference"]:.4f})
  Fidelity Penalty:      - {penalties['fidelity']:6.2f} (Fidelity: {m["fidelity"]:.3f})
  Depth Penalty:         - {penalties['depth']:6.2f} (Transpiled Depth: {m["transpiled_depth"]})
  CX Gate Penalty:       - {penalties['cx_gate']:6.2f} (CX Gates: {m["transpiled_cx_gates"]})
  Para Number Penalty    - {penalties['param_num']:6.2f} (Parameter Number: {m['num_params']})
  VQE Stability Penalty: - {penalties['stability']:6.2f} (StdDev: {m["energy_std_dev"]:.3f})
{extrapolation_section}
  -----------------------------------------------------------------
  Total Penalty:         - {total_penalty:6.2f}
--- VQE & CIRCUIT ANALYSIS ---
  VQE Minimum Energy:     {m["vqe_energy"]:.5f}
  True Ground Energy:     {m["true_ground_energy"]:.5f}
  Optimization Stability: {m["energy_std_dev"]:.5f} (Energy StdDev over {self.num_vqe_runs} runs)
  Final State Fidelity:   {m["fidelity"]:.4f}

--- CIRCUIT COMPLEXITY ---
  Parameters:             {m["num_params"]}
  Initial Depth:          {m["initial_depth"]} (Ops: {m.get('initial_ops', 'N/A')})
  Transpiled Depth:       {m["transpiled_depth"]} (Ops: {m.get('transpiled_ops', 'N/A')})
  Transpiled CX Gates:    {m["transpiled_cx_gates"]}

--- DIAGNOSIS & RECOMMENDATION ---
{diagnostic_summary}
=====================================================================
"""
        report = report_header + report_body
        return score, report


    def run(self):
        """Executes the full evaluation workflow."""
        try:
            self._prepare_environment()
            self._build_and_optimize_ansatz()
            self._run_vqe_simulations()
            self._run_extrapolation_check() # Check extrapolation performance
            self._analyze_and_transpile()
            return self._generate_report()
        except (ValueError, Exception) as e:
            print(f"\nEVALUATION FAILED: {e}")
            return 0.0, f"Evaluation Failed: {e}"

def _run_evaluation_process(q, idea, lamb, nsite_root, phi_max, num_vqe_runs, save_report):
    """Helper function to run the evaluation in a separate process and put the result in a queue."""
    try:
        evaluator = QuantumCircuitEvaluator(idea, lamb, nsite_root, phi_max, num_vqe_runs, save_report)
        score, report = evaluator.run()
        q.put((score, report))
    except (ValueError, Exception) as e:
        q.put((0.0, f"Evaluation Failed in subprocess: {e}"))

def evaluate(idea: str, lamb: float = 0.2, nsite_root: int = 3, phi_max: float = 0.45, num_vqe_runs: int = 60, save_report: bool = False, timeout: int = 1200):
    """
    Evaluates a quantum circuit idea by instantiating and running the QuantumCircuitEvaluator
    in a separate process with a timeout.
    
    Args:
        idea (str): A string containing the Python code for the 'create_ansatz' function.
        lamb (float): The interaction strength parameter for the Hamiltonian.
        nsite_root (int): The square root of the number of sites, defining the system size.
        num_vqe_runs (int): The number of VQE simulations to run with different initial points.
        save_report (bool): If True, saves the detailed report and visualizations to a directory.
        timeout (int): Timeout in seconds for the evaluation. If 0 or negative, no timeout.

    Returns:
        A tuple containing the final score (float) and a detailed report (str).
    """
    if timeout <= 0:
        # Run without timeout if timeout is not specified
        q = queue.Queue()
        _run_evaluation_process(q, idea, lamb, nsite_root, phi_max, num_vqe_runs, save_report)
        return q.get()

    q = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_evaluation_process,
        args=(q, idea, lamb, nsite_root, phi_max, num_vqe_runs, save_report)
    )
    
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join() # Ensure the process is fully terminated
        return 0.0, f"Evaluation Failed: Timed out after {timeout} seconds."
    
    try:
        return q.get_nowait()
    except queue.Empty:
        return 0.0, "Evaluation Failed: Process finished but no result was returned."

# test module
if __name__ == "__main__":
  # Example usage:
  # Ensure you have a valid idea file. You might need to adjust the path.
  idea_file_path = pathlib.Path(__file__).parent / 'database/ideas/initial_ideas/idea_uhocfz.idea'
  if idea_file_path.exists():
      with open(idea_file_path) as f:
          idea_code = f.read()
      # Running with a smaller number of VQE runs for a quick test.
      # For a full evaluation, use a higher number like 30 or more.
      # Set save_report=False to only print the report to the console.
      score, report = evaluate(idea_code, lamb=0.2, nsite_root=3, num_vqe_runs=60, save_report=True)
      print(f'\nFinal Score: {score:.2f}')
      print(report)
  else:
      print(f"Test idea file not found at: {idea_file_path}")
      print("Please update the path in the `if __name__ == '__main__'` block.")
