# Optimizing Gearbox Design

This repository focuses on optimizing gearbox design through advanced algorithmic approaches. It includes the implementation and testing of three different algorithms for solving high-dimensional optimization problems, with a specific focus on minimizing multivariable functions.

## Repository Structure

### 1. Algorithms for Optimizing Gearbox Design

#### `newway.py`
This script implements a novel ant colony optimization (ACO) algorithm designed for high-dimensional optimization problems, achieving improved accuracy and efficiency. 

##### Key Features of the Enhanced ACO Algorithm:
- **Domain Estimation and Transposition Operations**: These enhancements address slow convergence and the tendency to get stuck in local minima, common issues in traditional ACO.
- **Dynamic Learning Rates**: Ant movements are guided by learning rates that adapt based on convergence behavior, ensuring a balance between exploration and exploitation.
- **Check Class**: Monitors recent values and adjusts the learning rates. If the average gap between values drops below a certain threshold, the algorithm transitions to a focused search mode.
- **Moveback Feature**: Allows ants to revert to prior positions if the current step leads to less optimal results, reducing the risk of overshooting or becoming stuck in suboptimal areas.
- **Improved Convergence**: Dynamically adjusted learning rates help achieve superior performance on benchmark functions, including Ackley and Griewank, in 100-dimensional scenarios.

#### `normal-anti.py`
This script contains the implementation of a standard ant colony optimization algorithm. It serves as a baseline for performance comparison.

#### `improve-anti.py`
This script features an improved version of the standard ant colony optimization algorithm, incorporating enhancements to increase efficiency and accuracy in high-dimensional problems.

### 2. `Test_numerical.ipynb`
This Jupyter notebook tests the algorithms on the Ackley and Griewank functions in a 100-dimensional setting to evaluate their ability to find the global minimum of multivariable functions. 

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/tommyjr11/optimizing-gearbox-design.git
   cd optimizing-gearbox-design
   ```

2. Run the algorithms or the test notebook to evaluate performance:
   ```bash
   python newway.py
   python normal-anti.py
   python improve-anti.py
   ```

3. Use `Test_numerical.ipynb` to visualize and analyze the performance of the algorithms.

## Results
The `newway.py` algorithm demonstrated significant performance improvements over the baseline (`normal-anti.py`) and the improved version (`improve-anti.py`), particularly in high-dimensional benchmark tests using the Ackley and Griewank functions.

## Acknowledgments

Feel free to explore the repository and adapt the algorithms for your own high-dimensional optimization challenges!
