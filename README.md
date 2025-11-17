# 📈 Hessian Optimization Theory in Deep Learning

## Abstract

_(Content to be added here)_

---

## Introduction

_(Content to be added here)_

---

## Detail Description

### 1. Principles of Deep Learning

#### 1.1 Neural Networks and Non-linear Relations

A neural network consists of multiple layers, each containing several neurons. These neurons are connected through **weights** and **biases**. The connections between neurons allow the system to create a **non-linear mapping** from input to output.

The use of **activation functions** such as **ReLU**, **Sigmoid**, and **Tanh** is essential for enabling the network to form non-linear relationships. Multiple connected layers (**deep layers**) allow the model to learn highly complex patterns.

<p align="center">
  <img src="docs-src\images\neuron-network.png" alt="Neural Network Overview" width="450px">
  <br>
  <em>Figure 1: Structure and connections of artificial neurons (Source: GeeksforGeeks [1])</em>
</p>

---

### 1.2 Forward Pass Computation

A **_forward pass_** is the computation of the model’s output from the input, passing through all neurons and layers. The resulting prediction is then used to compute the **loss**, which will be utilized for gradient computation in the next section.

<p align="center">
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

## 2. First-Order Derivatives (Gradient)

### 2.1 Parameter Updates

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

### 2.2 Backward Pass (First-Order Derivative Computation)

The **backward pass** calculates the gradient of the loss function with respect to the parameters using the **Chain Rule**.

The first-order derivative of every parameter is computed by iterating backward from the final layer $L$ to the first layer.

$$
\displaystyle
z^{l} = a^{l-1} W^{l} + b^{l}, \quad
a^{l} = f(z^{l}), \quad
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

#### 2.2.1 Derivative of Loss w.r.t. Affine Input ($\delta^l$)

The **delta** ($\delta^l$) is the derivative of the loss with respect to the un-activated output ($z^l$).

$$\displaystyle \delta^{l} = \frac{\partial L}{\partial z^{l}} = \frac{\partial L}{\partial a^{l}} \cdot \frac{\partial a^{l}}{\partial z^{l}} = \frac{\partial \left( \frac{1}{n} \sum_{i=1}^{n} {(y_i - \hat{y}_i)^2} \right)}{\partial a^{l}} \cdot \frac{\partial f(z^{l})}{\partial z^{l}} = \frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \odot f'(z^{l})$$

#### 2.2.2 Gradient w.r.t. Weights and Biases (Chain Rule)

The gradients for the weights ($W^l$) and biases ($b^l$) are computed using $\delta^l$ and the activation from the previous layer:

$$
\displaystyle
\frac{\partial L}{\partial W^{l}} = (a^{l-1})^\top \delta^{l}, \quad
\frac{\partial L}{\partial b^{l}} = \delta^{l}
$$

#### 2.2.3 Backpropagating Delta to the Previous Layer

The delta is then propagated to the previous layer ($l-1$) to continue the backpropagation process:

$$
\displaystyle
\delta^{l-1} = \delta^{l} (W^{l})^\top \odot f'(z^{l-1})
$$

---

# 3. Working Principles of the Hessian in Deep Learning

## 2.1 Dimensions and Interpretation of the Hessian

- **Gradient (First-order derivative):** $\nabla L(\theta) \in \mathbb{R}^n$ ,Describes the slope of $L$

- **Hessian (Second-order derivative):** $\nabla^2 L(\theta) = H(\theta) \in \mathbb{R}^{n \times n}$ ,Describes the curvature of $L$

$$
\displaystyle
\boldsymbol{\theta}
=
\begin{bmatrix}
\theta_1
\\
\theta_2
\\
\vdots
\\
\theta_n
\end{bmatrix}

, \quad

\nabla L(\boldsymbol{\theta})
=
\begin{bmatrix}
\displaystyle \frac{\partial L}{\partial \theta_1}
\\
\displaystyle \frac{\partial L}{\partial \theta_2}
\\
\vdots
\\
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
$$

---

## 2.2 Components and Fundamental Properties of the Hessian

### 2.2.1 Clairaut's Theorem (Equality of Mixed Partials)

If mixed second partial derivatives are continuous over a region, then:

\[
H*{i,j} = H*{j,i}
\]

Thus, the Hessian is a **symmetric matrix**.

---

### 2.2.2 The Spectral Theorem

If the Hessian \( H(\theta) \) is a real symmetric matrix, it can be decomposed as:

\[
H = Q \Lambda Q^T
\]

Where:

#### 2.2.2.1 Λ (Lambda)

- A diagonal matrix
- Contains the eigenvalues \( \lambda_1, \lambda_2, ..., \lambda_n \) of \( H \)
- All eigenvalues are real numbers

#### Q (Orthogonal Matrix)

- Columns are the eigenvectors of \( H \)
- They form an orthonormal basis of \( \mathbb{R}^n \)
- \( Q^T = Q^{-1} \)
