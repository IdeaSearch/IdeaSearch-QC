# IdeaSearch Configuration for TF_and_XY Quantum Circuit Optimization

This document outlines the configuration for the `TF_and_XY_QuantumCircuit` project, which uses the IdeaSearch framework to optimize quantum circuit ansatz for simulating the ground state of Ising models.

## Project Information

-   **Project Name:** `TF_and_XY_QuantumCircuit`
-   **Description:** This project uses the IdeaSearch framework to automatically search and optimize quantum circuit ansatz for simulating the ground state of Ising models. By combining evolutionary algorithms with large language models, the system can automatically generate, evaluate, and optimize quantum circuit designs.
-   **Version:** `1.0.0`
-   **Author:** Sonny

## Model Selection

-   **Type:** `XY` (Options: `TF` for Transverse Field Ising, `XY` for XY Model)
-   **Hamiltonian Models:**
    -   **TF (Transverse Field Ising Model):** `H = -J * Σ Z_i Z_{i+1} - h * Σ X_i`
    -   **XY (XY Model):** `H = -J * Σ (X_i X_{i+1} + Y_i Y_{i+1}) - h * Σ Z_i`

## Paths Configuration

-   **API Keys:** `./api_keys.json` (Configure your API keys in this file)
-   **Database:** `./database` (Directory will be created automatically with a model type suffix)
-   **User Code:** `./user_code`

## Language Settings

-   **Interface Language:** `en_US` (Options: `zh_CN`, `en_US`)

## LLM Configuration

-   **Models:** `DeepSeek-R1-Distill-Qwen-32B`
-   **Temperatures:** `0.9`
-   **Examples in Prompt (SHOT_NUM):** `3`
-   **Sample Temperature for Examples (INIT_TEMP):** `10.0`

## Search Parameters

-   **Number of Islands (NUM_ISLANDS):** `10`
-   **Migration Cycles:** `3`
-   **Evolution Rounds per Cycle:** `10`
-   **Parallel Samplers per Island:** `3`
-   **Parallel Evaluators per Island:** `3`
-   **Ideas to Generate per Round (MAX_ATTEMPTS):** `3`

## Scoring Parameters

-   **Hand-over Threshold:** `0.0` (Minimum score to accept new ideas)
-   **Score Range:** `0.0` to `200.0`
-   **Scoring Mechanism:**
    -   **Raw Score:** VQE computed energy × (-10)
    -   **Length Penalty:** `min(2 * (code_length // 40) / 10, 2)`
    -   **Final Score:** `max(raw_score - length_penalty, 0.0)`
    -   **Parameter Limit:** Score is `0.0` if the parameter count exceeds 9.

## Evaluation Parameters

### Quantum System
-   **Number of Qubits:** `9`
-   **Coupling Strength (J):** `1.0`
-   **Field Strength (h):** `1.0`
-   **Periodic Boundary Condition (PBC):** `0` (0: open, 1: periodic)
-   **Maximum Parameters:** `9`

### VQE Optimizer
-   **Name:** `COBYLA`
-   **Max Iterations:** `3000`
-   **Rhobeg:** `300`
-   **Tolerance:** `1.0e-15`

### Timeout
-   **Evaluation Timeout:** `240.0` seconds

## Logging Configuration

-   **Record Prompts in Diary:** `true`
-   **Diary File:** `log/diary.txt` (Relative to the database path)
-   **Verbose:** `true`

## Initial Ideas Configuration

-   **Enabled:** `true`
-   **Count:** `3`
-   **Description:** Three initial quantum circuit templates are used:
    1.  Template 1: RX gates with linear parameter scaling
    2.  Template 2: RY gates with grouped parameters
    3.  Template 3: RZ gates with exponential parameters

## Workflow

1.  **Initialization:** Load three initial quantum circuit templates.
2.  **Sampling:** Select high-quality ideas from islands to use as examples.
3.  **Generation:** The LLM generates new quantum circuit code based on the examples.
4.  **Evaluation:** Evaluate the new circuit's performance using VQE.
5.  **Update:** Add high-scoring circuits to the island's population.
6.  **Migration:** Periodically exchange excellent ideas between islands.
7.  **Iteration:** Repeat the process until the specified number of cycles is complete.

## Output Configuration

-   **Subdirectories (auto-created):**
    -   `ideas`: Generated ideas
    -   `data`: Score data
    -   `pic`: Visualization charts
    -   `log`: Log files
-   **Display Settings:**
    -   Show best result per cycle: `true`
    -   Show final best result: `true`
    -   Show progress information: `true`

## Prerequisites

1.  **Install IdeaSearch:**
    ```bash
    pip install IdeaSearch
    ```
2.  **Install Dependencies:**
    ```bash
    pip install qiskit qiskit-aer qiskit-algorithms numpy scipy
    ```
3.  **Configure API Keys:**
    -   Ensure the `api_keys.json` file exists and is properly configured with API details for at least one model.

## Usage

1.  **Modify Configuration:**
    -   Edit the `config_en.yaml` file to set your desired parameters.
2.  **Run Search:**
    ```bash
    cd ideasearch_TF_and_XY
    python run_en.py
    ```
3.  **View Results:**
    -   **Real-time Output:** Check the terminal display.
    -   **Database Files:** Look in `database_TF/` or `database_XY/`.
    -   **Log Files:** See `database_*/log/diary.txt`.

## Notes

-   **Evaluation Time:** Each evaluation can take a significant amount of time (up to 240 seconds). Please be patient.
-   **Parameter Limit:** Generated circuits with more than 9 parameters will be rejected (score set to 0).
-   **Code Format:** Generated code must include a `create_ansatz(n)` function definition.
-   **Dependency Management:** Ensure all required Python packages are installed correctly.
-   **Thread Safety:** The evaluation function uses a thread-safe timeout mechanism, making it suitable for multi-threaded environments.

## Extensions

-   **Add More Models:** Add other LLMs to the `llm.models` list in the configuration.
-   **Custom Evaluation:** Modify `evaluation.py` to adapt to different physical systems.
-   **Mutation and Crossover:** Implement `mutation_func` and `crossover_func` to increase idea diversity.
-   **System Assessment:** Implement `assess_func` for global quality evaluation of the search process.

## References

-   [IdeaSearch Documentation](https://github.com/IdeaSearch/IdeaSearch)
-   [Qiskit Documentation](https://qiskit.org/documentation/)
-   [VQE Algorithm](https://qiskit.org/documentation/stubs/qiskit_algorithms.VQE.html)
-   **Transverse Field Ising Model:** `H = -J * Σ Z_i Z_{i+1} - h * Σ X_i`
-   **XY Model:** `H = -J * Σ (X_i X_{i+1} + Y_i Y_{i+1}) - h * Σ Z_i`