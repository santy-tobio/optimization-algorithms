# Numerical Optimization Methods Analysis

This project explores and implements various numerical optimization methods, analyzing their performance on both convex and non-convex problems. The study focuses on gradient descent methods and their variants, providing a comprehensive analysis of convergence properties and the effects of regularization.

## Project Overview

The analysis is conducted on two main problem types:
1. **Non-convex Optimization**: Implementation and analysis of optimization methods on the Rosenbrock function
2. **Convex Optimization**: Application to linear regression using the California Housing dataset

## Key Features

- Implementation of multiple optimization algorithms:
  - Gradient Descent
  - Newton-Raphson Method
  - Momentum-based Gradient Descent
  - Ridge Regression (L2 Regularization)

- Comprehensive analysis of:
  - Learning rate impact on convergence
  - Initial conditions sensitivity
  - Momentum parameter effects
  - Regularization parameter optimization
  - Numerical stability considerations

## Technical Implementation

### Core Components

- `funciones.py`: Contains implementations of optimization algorithms including:
  - Rosenbrock function and its derivatives
  - Gradient descent with various modifications
  - Newton-Raphson method
  - MSE and Ridge regression implementations

- Jupyter Notebooks:
  - Detailed experimental analysis
  - Visualization of results
  - Comparative performance studies

### Key Results

1. **Rosenbrock Function Optimization**:
   - Optimal learning rate identified at η = 3.73e-4
   - Newton-Raphson achieved quadratic convergence (10⁻¹⁶ precision in 5 iterations)
   - Analysis of numerical overflow conditions

2. **Linear Regression Analysis**:
   - Momentum acceleration reduced iterations by 82% (β = 0.95)
   - Ridge regularization optimization (λ = 9.10e-3)
   - Comparative analysis of exact vs. iterative methods

## Technical Highlights

- Implementation of numerically stable algorithms
- Rigorous theoretical analysis of convergence properties
- Comprehensive hyperparameter optimization
- Advanced numerical methods for handling ill-conditioned problems

## Tools and Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebooks

## Project Structure

```
.
├── Figures/ # plots in the report
├── codigos/
│   ├── rosenbrock_analysis.ipynb
│   └── linear_regression.ipynb
│   └── funciones.py              # Core implementations
└── report.pdf
└── README.md
```

## Results and Visualizations

The project includes detailed visualizations and analyses of:
- Convergence trajectories
- Learning rate sensitivity
- Momentum effects
- Regularization impact
- Error surface analysis

## Future Work

- Implementation of additional optimization algorithms
- Extension to stochastic gradient methods
- Analysis of alternative regularization techniques
- Application to deep learning optimization problems

## License

MIT License - see LICENSE file for details

---
## References

The theoretical foundation and implementation details are documented in the accompanying technical report. The project builds upon classical optimization theory and modern numerical methods.
