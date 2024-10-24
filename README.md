# FEOHO-NN: Neural Network in Rust
2 | 2:47:46

## Naming of library:

- `feoho` is $Fe_2O_3nH_2O$ i.e. `Rust`
- `nn` is Neural network

## Math behind basic neural network

<!--$f(x) = y = x * w + b$-->

Output function:
$$f(x) = y = x * w + b$$

- Finite Difference:
  ```math
  L = \lim_{x \rightarrow 0}{\frac{f(a + h) - f(a)}{h}}
  ```

## Neural Network Activation Functions

### 1. **Tanh (Hyperbolic Tangent)**
$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$
- **Range**: (-1, 1)
- **Use Case**: Often used in feedforward and recurrent neural networks, especially where zero-centered outputs are desired. It helps mitigate the vanishing gradient problem better than the Sigmoid function.

---

### 2. **Softmax**
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j}e^{x_j}}
$$
- **Range**: (0, 1)
- **Use Case**: Typically used in the output layer for multi-class classification problems. It converts raw class scores into probabilities.

---

### 3. **Sigmoid**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
- **Range**: (0, 1)
- **Use Case**: Commonly used for binary classification problems. It squashes the input into a range between 0 and 1, though it's prone to vanishing gradient problems.

---

### 4. **ReLU (Rectified Linear Unit)**
$$
\text{ReLU}(x) = \max(0, x)
$$
- **Range**: [0, ∞)
- **Use Case**: Widely used in hidden layers of deep learning networks due to its simplicity and effectiveness. Helps mitigate the vanishing gradient problem by allowing gradients to propagate when activations are positive.

---

### 5. **Leaky ReLU**
$$
\text{Leaky ReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{otherwise}
\end{cases}
$$
- **Range**: (-∞, ∞)
- **Use Case**: Addresses the dying ReLU problem by allowing a small, non-zero gradient when the unit is inactive (i.e., when `x < 0`).

---

### 6. **ELU (Exponential Linear Unit)**
$$
\text{ELU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{otherwise}
\end{cases}
$$
- **Range**: (-α, ∞)
- **Use Case**: Similar to Leaky ReLU, but with smoother transitions. Often used to speed up learning and allow for more robust convergence in deep networks.

---

### 7. **Swish**
$$
\text{Swish}(x) = x \cdot \sigma(x)
$$
- **Range**: (-∞, ∞)
- **Use Case**: Found to outperform ReLU in some architectures and is smoother than both ReLU and Leaky ReLU. Often used in deep networks.

---

### 8. **Maxout**
$$
\text{Maxout}(x_1, x_2, \dots, x_n) = \max(x_1, x_2, \dots, x_n)
$$
- **Range**: (-∞, ∞)
- **Use Case**: More flexible than ReLU, it can approximate a wide variety of activation functions. Often used with dropout for regularization.

---

### 9. **Softplus**
$$
\text{Softplus}(x) = \ln(1 + e^x)
$$
- **Range**: (0, ∞)
- **Use Case**: A smooth approximation to ReLU, it is often used in scenarios where a smoother activation function is required.

---

### 10. **Softsign**
$$
\text{Softsign}(x) = \frac{x}{1 + |x|}
$$
- **Range**: (-1, 1)
- **Use Case**: Similar to Tanh, but with a gentler gradient. It's sometimes used in place of Tanh due to its smoother transitions.

---

### 11. **Logistic Sigmoid**
$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$
- **Range**: (0, 1)
- **Use Case**: Used primarily in binary classification problems and binary logistic regression. However, it is susceptible to vanishing gradients.

---

### 12. **Linear Activation Function**
$$
f(x) = x
$$
- **Range**: (-∞, ∞)
- **Use Case**: Typically used in the output layer for regression tasks, where the predicted output can take on any real value.

---

### 13. **Step Function**
$$
f(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
$$
- **Range**: {0, 1}
- **Use Case**: Used in decision-making or classification processes but is rarely used in modern neural networks due to its non-differentiability.

---

### 14. **Softmax**
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$
- **Range**: (0, 1)
- **Use Case**: Often used in the output layer for multi-class classification problems to compute class probabilities.

