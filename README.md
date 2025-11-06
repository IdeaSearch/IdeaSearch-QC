# Quantum Circuit Design Repository

This repository contains quantum circuit design tools for lattice field theory simulations using variational quantum algorithms.

## Chapter 1: Preparation

### Prerequisites

Before using this repository, you need to set up the required dependencies and configurations.

#### 1.1 Install IdeaSearch Framework

First, install the IdeaSearch package:

```bash
pip install IdeaSearch
```

For detailed documentation and setup instructions, visit the official repository:
**https://github.com/IdeaSearch/IdeaSearch-framework**

#### 1.2 Configure API Keys

Follow the instructions in the IdeaSearch package documentation to create your API keys JSON file. This file is required to access AI providers (such as OpenAI, Google Gemini, DeepSeek, etc.) for automated circuit design.

The API keys JSON file should contain your credentials for the AI services you plan to use. Refer to the IdeaSearch documentation for the exact format and supported providers.

#### 1.3 Install Qiskit Dependencies

Install the specific versions of Qiskit and Qiskit Algorithms required for this project:

```bash
pip install qiskit==1.4.2
pip install qiskit_algorithms==0.3.1
```

**Important:** These specific versions are required for compatibility. Using different versions may cause errors.

## Chapter 2: TF and XY

See the detailed instructions in the [ideasearch_TF_and_XY README](ideasearch_TF_and_XY/readme.md) for setting up and running the quantum circuit design for the TF and XY models.

## Chapter 3: Scalar Field Theory

This section covers the quantum circuit design for scalar field theory on lattice systems.

### 3.1 Configuration

Before running the circuit design optimization, you need to configure the paths in the main script:

1. Open the file `qcRepo/scalar/optimize/run_circuit_design.py`

2. Modify the following paths according to your setup:

```python
# Replace these paths with your own
api_path = "Your api key position"  # Path to your API keys JSON file
data_path = "Your database path"    # Path where results will be saved
```

**Example:**
```python
api_path = "/home/username/api_keys.json"
data_path = "/home/username/quantum_circuits/database"
```

### 3.2 Running the Circuit Design Optimization

Once you have configured the paths and installed all dependencies:

1. Navigate to the scalar optimize directory:
```bash
cd qcRepo/scalar/optimize
```

2. Run the circuit design script with the correct Python environment:
```bash
python run_circuit_design.py
```

The script will:
- Load AI models specified in the configuration
- Generate novel quantum circuit ansatz designs
- Evaluate each design using VQE (Variational Quantum Eigensolver)
- Optimize circuits across multiple islands using evolutionary algorithms
- Save the best performing circuits to your specified data path

### 3.3 Viewing Results

After running the optimization, you can find the optimized circuit codes in the database path you configured (`data_path`). The results include:

- **Circuit designs**: Python code for the best performing ansatz
- **Performance metrics**: Fidelity, energy, circuit depth, and other evaluation scores
- **Evolution history**: Track how circuits improved over iterations

Check the subdirectories in your `data_path` for detailed results and analysis reports.

### 3.4 Understanding the System

The scalar field theory circuit design system includes:

- **Symmetry-aware building blocks**: Pre-defined functions that enforce translational and C4 rotational symmetry
- **Automated evaluation**: VQE-based scoring system that measures circuit performance
- **Multi-model AI generation**: Uses multiple AI models with different temperatures for diverse exploration
- **Island-based evolution**: Parallel optimization across multiple populations with periodic mixing

For more details on the physics and methodology, refer to the code documentation in the source files.

---

## Additional Resources

- IdeaSearch Framework: https://github.com/IdeaSearch/IdeaSearch-framework
- For questions or issues, please refer to the respective package documentation

## License

*License information to be added*
