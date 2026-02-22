<!-- filename: reparametrization-validation.md -->
To validate reparametrization strategies (affine and nonlinear) and ensure gradient consistency, follow these steps with precise implementation details:

---
### **1. Affine and Nonlinear Reparametrization**
- **Affine Transformation**: Scale and shift parameters.
  <code>
  def affine_reparam(p, scale=1.0, shift=0.0):
      return p * scale + shift
  </code>
- **Nonlinear Transformation**: Use a function like tanh for non-linearity.
  <code>
  def nonlinear_reparam(p):
      return torch.tanh(p)
  </code>

---
### **2. Gradient Computation**
Use PyTorch's `torch.autograd.grad` to compute gradients. Ensure the model and reparametrized parameters are correctly linked.

<code>
def compute_gradients(model, params, inputs, loss_func):
    params.requires_grad_(True)
    outputs = model(inputs)
    loss = loss_func(outputs, targets)
    grads = torch.autograd.grad(loss, params, retain_graph=True)
    return grads
</code>

---
### **3. Jacobian Computation**
Compute the Jacobian of the reparametrization function using `torch.autograd.functional.jacobian`.

<code>
def compute_jacobian(reparam_func, params):
    params.requires_grad_(True)
    output = reparam_func(params)
    jacobian = torch.autograd.functional.jacobian(reparam_func, params)
    return jacobian
</code>

---
### **4. Invariance Test (Gradient Consistency)**
Verify that gradients under reparametrization match the original gradients via the chain rule.

**Correct Chain Rule Relationship**:
- Let $ q = f(p) $, then $ \frac{dL}{dp} = \frac{dL}{dq} \cdot \frac{dq}{dp} $.
- **Test**: Check if `original_grads ≈ reparam_grads @ jacobian`.

<code>
def test_invariance(original_grads, reparam_grads, jacobian):
    # Reshape gradients for matrix multiplication
    original_grads = original_grads.reshape(-1, 1)
    reparam_grads = reparam_grads.reshape(1, -1)
    jacobian = jacobian.reshape(-1, 1)
    # Compute expected gradients
    expected = reparam_grads @ jacobian
    # Check if original gradients match expected
    return torch.allclose(original_grads, expected, atol=1e-6)
</code>

---
### **5. Validation Workflow**
<code>
# Example validation workflow
params = ...  # Original parameters
reparam_params = nonlinear_reparam(params)

# Compute gradients for original parameters
original_grads = compute_gradients(model, params, inputs, loss_func)

# Compute gradients for reparametrized parameters
reparam_grads = compute_gradients(model, reparam_params, inputs, loss_func)

# Compute Jacobian
jacobian = compute_jacobian(nonlinear_reparam, params)

# Test invariance
assert test_invariance(original_grads, reparam_grads, jacobian)
</code>

---
### **Key Takeaways**
- **Chain Rule**: Ensure gradients align with $ \frac{dL}{dp} = \frac{dL}{dq} \cdot \frac{dq}{dp} $.
- **Efficiency**: Use sparse operations or approximations for scalability.
- **Validation**: Combine analytical and numerical methods for robustness.