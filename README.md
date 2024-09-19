# FBSTD: Forward-Backward Stochastic equations based Temporal Difference method

This repository contains the implementation of the Forward-Backward Stochastic equations based Temporal Difference (FBSTD) method as described in the paper "[Deep Neural Networks Based Temporal-Difference Methods for High-Dimensional Parabolic Partial Differential Equations](https://www.sciencedirect.com/science/article/abs/pii/S0021999122005654)."

## Overview

Solving high-dimensional partial differential equations (PDEs) is a challenging task due to the curse of dimensionality that classical numerical methods face. This repository implements a novel approach using deep neural networks (NN) based temporal-difference (TD) learning methods to solve these high-dimensional parabolic PDEs efficiently.

The key idea is to approximate the solution of the PDE using a neural network function. The original deterministic parabolic PDE is transformed into a forward-backward stochastic differential equations (FBSDE) system using the nonlinear Feynman-Kac formula. This system is then further transformed into a Markov reward process (MRP), which allows the use of reinforcement learning techniques to train the neural network function.

## Features

- **High-dimensional PDEs**: Efficiently solve PDEs in high-dimensional spaces, even with dimensions as high as 100.
- **Deep Learning Integration**: Utilizes neural networks to approximate solutions, accelerating computational speed and improving accuracy.
- **Temporal-Difference Learning**: Implements a temporal-difference learning framework with reinforcement learning techniques.

## Citations

If you use this code in your research, please cite the original paper:

```
@article{zeng2022deep,
  title={Deep neural networks based temporal-difference methods for high-dimensional parabolic partial differential equations},
  author={Zeng, Shaojie and Cai, Yihua and Zou, Qingsong},
  journal={Journal of Computational Physics},
  volume={468},
  pages={111503},
  year={2022},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
