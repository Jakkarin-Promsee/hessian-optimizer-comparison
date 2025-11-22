# Discretization of Gradient Flow

## Overview

This repository explores and compares three gradient descent approaches—**Explicit GD**, **Implicit GD**, and **Newton GD**—using only **NumPy** for maximum transparency and control over the gradient flow. The goal is to study their convergence behaviors in depth.

### Models Explanation

- **Explicit GD (Standard Gradient Descent)**

  - **Formula:**

$$\theta_{k+1} = \theta_k - \eta \nabla L_k$$

- **Explanation:**  
  Uses a **first-order (linear) Taylor approximation** of the loss to update parameters. Each step moves along the gradient direction with step size $\eta$. Simple but may struggle with stiff or ill-conditioned loss landscapes.

- **Implicit GD (Backward Euler on Quadratic)**

  - **Formula:**

$$\theta_{k+1} = (I + \eta H_k)^{-1} \left(\theta_k + \eta (H_k \theta_k - \nabla L_k)\right)$$

- **Explanation:**  
  Uses a **second-order (quadratic) Taylor approximation**, effectively taking the curvature of the loss into account. This allows more stable updates, particularly in stiff regions, by solving for the next step implicitly.

- **Newton GD (Newton’s Method)**
  - **Formula:**

$$\theta_{k+1} = \theta_k - H_k^{-1} \nabla L_k$$

- **Explanation:**  
  Also a quadratic approximation, but it **directly jumps to the local optimum** of the quadratic approximation in a single step.
  - **Note:** Damping and Armijo line search are used to prevent instability, as Newton updates can diverge if the Hessian is ill-conditioned or the step overshoots.

### Experiment Objective

- Compare the **advantages and disadvantages** of each gradient descent method.
- Analyze convergence speed and stability using **validation loss** curves over training batches.

### Experiment Results

#### 1. Early Convergence (Initial Steps)

- Observed speed:

$$\text{Newton GD} > \text{Implicit GD} > \text{Explicit GD}$$

- **Reason:**
  - Newton GD can jump directly to the optimum of the local quadratic, achieving rapid initial convergence.
  - Implicit GD moves along a precise quadratic path, slower than Newton but faster than explicit GD.
  - Explicit GD follows a rough linear path, making its early convergence the slowest.

<p align="center" style="text-align: center;">
  <img src="docs-src/result/short-convergence-speed.png" alt="Early Convergence Comparison" width="450px">
  <br>
  <em>Figure 1: Validation loss comparison (batch 0–45, ~0–1 epochs)</em>
</p>

#### 2. Late Convergence (Long-Term Behavior)

- Observed trend:

$$\text{Implicit GD} > \text{Explicit GD}, \quad \text{Newton GD is unstable}$$

- **Insights:**
  - Implicit GD’s quadratic approximation ensures fast and **stable convergence**, often faster than explicit GD in the long term. Convergence resembles $\frac{1}{x^2}$ versus $\frac{1}{x}$ for explicit GD.
  - Explicit GD eventually converges but requires much longer training.
  - Newton GD becomes unstable when the loss drops below ~4%, as Hessian sensitivity amplifies noise, causing erratic updates.

#### 3. Conclusion (Long-Term Behavior)

- **Explicit GD**: Slow but safe; eventually converges with enough training.
- **Implicit GD**: Stable and follows a quadratic path; converges faster than explicit GD in the long term.
- **Newton GD**: Extremely fast at the beginning but becomes unstable later due to Hessian sensitivity and noise amplification.

<p align="center" style="text-align: center;">
  <img src="docs-src/result/long-convergence-speed.png" alt="Long-Term Convergence Comparison" width="450px">
  <br>
  <em>Figure 2: Validation loss comparison over extended training</em>
</p>

## Project Structure

```bash
Project/
├── Custom_library/
│    │
│    ├── core/               # The absolute fundamentals of all models
│    │   ├── baseLrModels.py             # base models funciton: getHistory(), predict(), save(), etc.
│    │   └── baseLrModelsImplementHessain.py         # implement compute tools: computeLoss(), computeGradient(), computeHessain()
│    │
│    ├── layers/             # Pure layer implementations
│    │   ├── denseLayer.py
│    │   └── activationLayer.py
│    │
│    ├── models/             # Training *strategies*
│    │   ├── explicitLrModel.py
│    │   ├── implicitLrModel.py
│    │   ├── newtonLrModel.py
│    │   └── optimizeNewtonLrModel.py
│    │
│    └── utils/              # Tools, not ML logic
│        ├── history.py
│        ├── metrics.py
│        └── dataUtils.py
│
├── main.ipynb      # Experiment lab, including all process
│
└── README.md       # Experiment report, including all description
```

## Full Proof

### 1. Principles of Deep Learning

#### 1.1 Neural Networks and Non-linear Relations

A neural network consists of multiple layers, each containing several neurons. These neurons are connected through **weights** and **biases**. The connections between neurons allow the system to create a **non-linear mapping** from input to output.

The use of **activation functions** such as **ReLU**, **Sigmoid**, and **Tanh** is essential for enabling the network to form non-linear relationships. Multiple connected layers (**deep layers**) allow the model to learn highly complex patterns.

<p align="center"  style="text-align: center;">
  <img src="docs-src\images\neuron-network.png" alt="Neural Network Overview" width="450px">
  <br>
  <em>Figure 1: Structure and connections of artificial neurons (Source: GeeksforGeeks [1])</em>
</p>

---

### 1.2 Forward Pass Computation

A **_forward pass_** is the computation of the model’s output from the input, passing through all neurons and layers. The resulting prediction is then used to compute the **loss**, which will be utilized for gradient computation in the next section.

<p align="center"  style="text-align: center;">
  <img src="docs-src\images\singular-neural.png" alt="Neural Network Overview" width="250px">
  <br>
  <em>Figure 2: Computation within a singular artificial neuron (Source: GeeksforGeeks [1])</em>
</p>

#### 1.2.1 Affine Transformation in Each Layer

For each layer $l$, the output is computed using a linear (affine) transformation:

$$\displaystyle z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$$

where:

- $\displaystyle a^{(l-1)}$ : activation from the **previous layer**
- $\displaystyle W^{(l)}$ : **weight matrix**
- $\displaystyle b^{(l)}$ : **bias vector**

#### 1.2.2 Activation Computation

The activation of the current layer is then computed as:

$$\displaystyle a^{(l)} = f(z^{(l)})$$

where $f(x)$ is an **activation function** such as ReLU, Sigmoid, or Tanh.

#### 1.2.3 Recursive Computation Through All Layers

The computation is repeated from:

$$\displaystyle a^{(0)} = \text{input vector}$$

until the last layer $L$, $a^{(L)}$, which is the model's final prediction. We denote this last answer as $y_i$.

#### 1.2.4 Loss Computation

$$\displaystyle L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Typically, **Mean Square Error (MSE)** is used to compute the prediction error. We perform forward computation for $N$ inputs in a batch, and the error is averaged to obtain a scalar loss value.

---

### 1.3 First-Order Derivatives (Gradient)

#### 1.3.1 Parameter Updates

The idea is to use the **slope of the function** (first-order derivative or **gradient**) to determine the correct direction for updating parameters.

$$
\displaystyle
\theta = \theta - \eta \nabla L, \quad
W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}, \quad
b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
$$

**Example:**
If the gradient of the loss $L$ with respect to $W^{(l)}$ is 2, it means:

- If we increase $W^{(l)}$ by 1 unit, the loss $L$ increases by 2 units.

To reduce the error, we adjust $W^{(l)}$ in the **_opposite direction of the gradient_**, scaled by the **learning rate** $\eta$.

#### 1.3.2 Backward Pass (First-Order Derivative Computation)

The **backward pass** calculates the gradient of the loss function with respect to the parameters using the **Chain Rule**.

The first-order derivative of every parameter is computed by iterating backward from the final layer $L$ to the first layer.

$$
\displaystyle
z^{l} = a^{l-1} W^{l} + b^{l}, \quad
a^{l} = f(z^{l}), \quad
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

##### 1.3.2.1 Derivative of Loss w.r.t. Affine Input ($\delta^l$)

The **delta** ($\delta^l$) is the derivative of the loss with respect to the un-activated output ($z^l$).

$$\displaystyle \delta^{l} = \frac{\partial L}{\partial z^{l}} = \frac{\partial L}{\partial a^{l}} \cdot \frac{\partial a^{l}}{\partial z^{l}} = \frac{\partial \left( \frac{1}{n} \sum_{i=1}^{n} {(y_i - \hat{y}_i)^2} \right)}{\partial a^{l}} \cdot \frac{\partial f(z^{l})}{\partial z^{l}} = \frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \odot f'(z^{l})$$

##### 1.3.2.2 Gradient w.r.t. Weights and Biases (Chain Rule)

The gradients for the weights ($W^l$) and biases ($b^l$) are computed using $\delta^l$ and the activation from the previous layer:

$$
\displaystyle
\frac{\partial L}{\partial W^{l}} = (a^{l-1})^\top \delta^{l}, \quad
\frac{\partial L}{\partial b^{l}} = \delta^{l}
$$

##### 1.3.2.3 Backpropagating Delta to the Previous Layer

The delta is then propagated to the previous layer ($l-1$) to continue the backpropagation process:

$$
\displaystyle
\delta^{l-1} = \delta^{l} (W^{l})^\top \odot f'(z^{l-1})
$$

---

# 2. Components and Fundamental Properties of the Hessian

## 2.1 Dimensions and Interpretation of the Hessian

- **Gradient (First-order derivative):** $\nabla L(\theta) \in \mathbb{R}^n$ ,Describes the slope of $L$

- **Hessian (Second-order derivative):** $\nabla^2 L(\theta) = H(\theta) \in \mathbb{R}^{n \times n}$ ,Describes the curvature of $L$

<p align="center"  style="text-align: center;">
  <img src="docs-src\latex\2.1.hessain-dimension.png" alt="Neural Network Overview" width="600px">
</p>

<!-- $$
\displaystyle
\boldsymbol{\theta}
=
\begin{bmatrix}
\theta_1 \\ \theta_2 \\ \vdots \\ \theta_n
\end{bmatrix}

, \quad

\nabla L(\boldsymbol{\theta})
=
\begin{bmatrix}
\displaystyle \frac{\partial L}{\partial \theta_1}
\\[10pt]

\displaystyle \frac{\partial L}{\partial \theta_2}
\\[10pt]

\vdots
\\[10pt]

\displaystyle \frac{\partial L}{\partial \theta_n}
\end{bmatrix}

, \quad

H(\boldsymbol{\theta})
=
\nabla^2 L(\boldsymbol{\theta})
=
\begin{bmatrix}
\displaystyle \frac{\partial^2 L}{\partial \theta_1^2}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_1 \partial \theta_2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_1 \partial \theta_d}
\\[10pt]

\displaystyle \frac{\partial^2 L}{\partial \theta_2 \partial \theta_1}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_2^2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_2 \partial \theta_d}
\\[10pt]

\vdots & \vdots & \ddots & \vdots
\\[10pt]

\displaystyle \frac{\partial^2 L}{\partial \theta_d \partial \theta_1}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_d \partial \theta_2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_d^2}
\end{bmatrix}
$$ -->

### 2.2 Sysmetrix Properties

From Clairaut's Theorem (Equality of Mixed Partials), If mixed second partial derivatives are continuous over a region, then:

$$
\frac{\partial^2 f(x)} {\partial x_i \partial x_j}
= \frac{\partial^2 f(x)} {\partial x_j \partial x_i} \quad
\longrightarrow \quad
H_{ij}(\theta)
= \frac{\partial^2 f(x)} {\partial \theta_i \partial \theta_j}
= \frac{\partial^2 f(x)} {\partial \theta_j \partial \theta_i}
= H_{ji}(\theta)
$$

### 2.3 Quadratic Form

The quadratic form $Q(\mathbf{x}) = \mathbf{x}^T H \mathbf{x}$ is the second-order term of a function's Taylor expansion. It tells us about the local curvature (or shape) of the function $f(\mathbf{x})$ around a critical point.

$$
\begin{align*}
Q(\mathbf{x}) &= \mathbf{x}^T \mathbf{H} \mathbf{x} \in \mathbb{R}, \quad \forall \mathbf{x} \in \mathbb{R}^n \quad \text{(The Quadratic Form)} \\
&= \sum_{i=1}^n \sum_{j=1}^n h_{ij} x_i x_j \\
&= \sum_{i=1}^n h_{ii} x_i^2 + \sum_{i \ne j}^n h_{ij} x_i x_j \\
&= \sum_{i=1}^n h_{ii} x_i^2 + 2 \sum_{i \lt j}^n h_{ij} x_i x_j \in \mathbb{R} \quad \text{, where } \mathbf{H} \text{ is symmetric}
\end{align*}
$$

- If Q(x) > 0 (Positive Definite), the function is convex and the critical point is a local minimum.
- If Q(x) < 0 (Negative Definite), the function is concave and the critical point is a local maximum.
- If Q(x) takes on both signs (Indefinite), the critical point is a saddle point.

### 2.4 Eigen-decomposit

From the Spectral Theorem, if $H \in \mathbb{R}^{n \times n}$ and $H$ is symmetric ($H = H^T$), then:

$$
\lambda_i \in \mathbb{R}, \quad \text{for all } i = 1, \ldots, n ,\quad
\mathbf{v}_i \in \mathbb{R}^n, \quad \text{for all } i = 1, \ldots, n
$$

Proving:

1. $A \in \mathbb{R}^{n \times n}, \lambda \in \mathbb{R}, v \in \mathbb{R}^2$ that $Av = \lambda v \ (1)$
2. take (1) with $v^\dagger$ from both left size

$$
\begin{array}{l}
v^\dagger (Av) = v^\dagger (\lambda v) \quad
(v^\dagger A) v = \lambda (v^\dagger v) \quad (2)
\end{array}
$$

3. take (2) with conjugate transpose

$$
[(v^\dagger A) v]^\dagger = v^\dagger A^\dagger v^{\dagger\dagger} = v^\dagger A^\dagger v \quad (3)
$$

4. Because $A \in \mathbb{R}^{n \times n}$, making $\bar{A} = A$. And because $A$ is a symmetric, making $A^T = A$. So from (3):

$$
\begin{array}{l}
v^\dagger A^\dagger v = v^\dagger \bar{A}^T v = v^\dagger A v \quad (4) \\
v^\dagger A^\dagger v = [\lambda (v^\dagger v)]^\dagger = \bar{\lambda} (v^\dagger v) \quad (5)
\end{array}
$$

5. From (2) is equal (5):

$$
(v^\dagger A) v = \lambda (v^\dagger v) = \bar{\lambda} (v^\dagger v)
\\
(\lambda - \bar{\lambda})(v^\dagger v) = 0
$$

6. Because $v^\dagger v \ne 0$, So:

$$
(\lambda - \bar{\lambda}) = 0 \quad \rightarrow \quad
\lambda = \bar{\lambda} \quad \longrightarrow \quad
\forall \lambda \in \mathbb{R}
$$

7. Using Eigen vector formular:

$$
(A - \lambda I)v = 0
$$

We know $A \in \mathbb{R}^{n \times n}, \lambda \in \mathbb{R}$, Thus $\forall (A-\lambda I) \in \mathbb{R}$, Thus $\forall v \in \mathbb{R}$.

### 2.5 Orthogonal of Eigen vector

From the Spectral Theorem, if $H \in \mathbb{R}^{n \times n}$ and $H$ is symmetric ($H = H^T$), then eigen vector of $H$ will be orthogonal.

Proving:

1. Consider $Av_1 = \lambda_1 v_1 \ (1)$ and $Av_2 = \lambda_2 v_2 \ (2)$
2. Start with $v_1^T (A v_2)$ to substitude $A v_2$:

$$
\begin{array}{l}
v_1^T (A v_2) = v_1^T (\lambda_2 v_2) \quad \\
v_1^T (A v_2) = \lambda_2 (v_1^T v_2) \quad (3)
\end{array}
$$

3. Since $v^T (A v)$ is sclale 1 x 1, $v^T (A v) = [v^T (A v)]^T$. From (3):

$$
\begin{align*}
(A v_2) &= [v_1^T (A v_2)]^T \\
&= v_2^T A^T v_1 \quad, A \ \text{is symmetric} \\
&= v_2^T A v_1 \quad, \text{substitude (1)} \\
&= v_2^T (\lambda_1 v_1) \\
&= \lambda_1 (v_2^T v_1) \quad (4)
\end{align*}
$$

4. (3) is equal (4):

$$
\begin{array}{c}
\lambda_1 (v_2^T v_1) = \lambda_2 (v_1^T v_2) \\
(\lambda_1-\lambda_2)(v_1^T v_2) = 0
\end{array}
$$

5. As spectral's theorem, we know $\lambda_i \in \mathbb{R}, \quad \text{for all } i = 1, \ldots, n ,\quad \mathbf{v}_i \in \mathbb{R}^n, \quad \text{for all } i = 1, \ldots, n$. So $\lambda_1 \ne \lambda_2$, making $(\lambda_1 \ne \lambda_2)$ is non-zero. Thus, $(v_1^T v_2) = 0$ or $v_1 \dot v_2$ = 0. Thus $v_1$ and $v_1$ is orthogonal.

### 2.6 Quadratic Form of symmetric Hessain for Eigen-decomposit

The Hessian matrix, $\mathbf{H}$, being symmetric ($\mathbf{H} = \mathbf{H}^T$), can be decomposed via the Spectral Theorem as $\mathbf{H} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$. In this decomposition, the eigenvalues ($\lambda_i$) are guaranteed to be real numbers, and the corresponding eigenvectors ($\mathbf{v}_i$), collected in the orthogonal matrix $\mathbf{Q}$, form a basis of principal directions.

Applying this decomposition to the quadratic form $Q(\mathbf{x}) = \mathbf{x}^T \mathbf{H} \mathbf{x}$ allows for a precise analysis of the local curvature. The eigenvalues $\lambda_i$ quantify the magnitude and sign of the curvature along the unique directions defined by the eigenvectors $\mathbf{v}_i$.

$$
\begin{align*}
Q(x) = x^T H x &= x^T Q \Lambda Q^T x \\
&= (Q^T x)^T \Lambda (Q^T x) \\
&= \sum_{i}^n \Lambda_{ii} (Q^T x)_i^2 + 2 \sum_{i<j}^n \Lambda_{ii} (Q^T x)\_i (Q^T x)\_j \\
&= \sum_{i}^n \lambda\_{ii} (v_i^T x)^2
\end{align*}
$$

- $\lambda_i$ = Priciple Maginitude.
- $v_i$ = Priciple Direction.
- $(v_i^T x)^2$ = Contribution of $x$ in $v_i$ direction.

---

# On progress

---

# Prop

$$

\begin{array}{l}

\end{array}


$$

$$
$$
