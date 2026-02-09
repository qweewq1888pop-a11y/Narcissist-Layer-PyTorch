import torch

import torch.nn as nn

from torch.autograd import Function



# ==========================================

#  1. THE FUNCTIONAL ENGINE (Mathematical Core)

# ==========================================



class NarcissistFunction(Function):

    """

    Autograd Function for the Narcissist Operation:

    y = x * sigmoid( w * (x - mean(x)) )

    """

    

    @staticmethod

    def forward(ctx, x, w, dim=None):

        # 1. Statistics

        if dim is None:

            mu = x.mean()

        else:

            mu = x.mean(dim=dim, keepdim=True)

            

        # 2. Deviation & Gating

        diff = x - mu

        z = diff * w

        gate = torch.sigmoid(z)

        y = x * gate



        # 3. Context Saving

        # We perform the '1 - gate' calculation here if memory permits, 

        # or recompute in backward to save VRAM. Here we save tensors.

        ctx.save_for_backward(x, w, gate, diff)

        ctx.dim = dim

        ctx.input_shape = x.shape

        

        return y



    @staticmethod

    def backward(ctx, grad_output):

        x, w, gate, diff = ctx.saved_tensors

        dim = ctx.dim

        input_shape = ctx.input_shape



        grad_input = grad_weight = None



        # --- PRE-CALCULATION ---

        # d(gate)/d(z) = gate * (1 - gate)

        d_gate_d_z = gate * (1.0 - gate)

        

        # dL/dz = (dL/dy * x) * d(gate)/dz

        # This represents the gradient flowing into the "Urge" (z)

        grad_z = grad_output * x * d_gate_d_z



        # --- GRADIENT W.R.T INPUT (x) ---

        if ctx.needs_input_grad[0]:

            # Path A: Direct path through the multiplication (y = x * gate)

            # dL/dx_direct = grad_output * gate

            term_direct = grad_output * gate



            # Path B: Path through the deviation/mean (z = w * (x - mu))

            # 1. Backprop through scaling by w:  grad_diff = grad_z * w

            grad_diff = grad_z * w

            

            # 2. Backprop through mean subtraction (x - mu):

            # The gradient of (x - mean(x)) is (g - mean(g))

            if dim is None:

                term_mean = grad_diff - grad_diff.mean()

            else:

                term_mean = grad_diff - grad_diff.mean(dim=dim, keepdim=True)



            grad_input = term_direct + term_mean



        # --- GRADIENT W.R.T WEIGHT (w) ---

        if ctx.needs_input_grad[1]:

            # dL/dw = dL/dz * dz/dw

            # dz/dw = diff

            raw_grad_w = grad_z * diff



            # Handle Broadcasting: Sum gradients over dimensions where w was broadcasted

            # We compare w.shape to x.shape to find which dims need summing

            if w.shape != input_shape:

                # 1. Identify dims that exist in x but not w (or are 1 in w)

                ndim_x = len(input_shape)

                ndim_w = w.ndim

                

                # Prepend 1s to w shape to match x rank

                w_shape_expanded = (1,) * (ndim_x - ndim_w) + w.shape

                

                dims_to_sum = []

                for i, (dim_x, dim_w) in enumerate(zip(input_shape, w_shape_expanded)):

                    if dim_x != dim_w:

                        dims_to_sum.append(i)

                

                if dims_to_sum:

                    grad_weight = raw_grad_w.sum(dim=dims_to_sum, keepdim=True)

                    

                # Reshape back to original w shape (removes the prepended 1s)

                grad_weight = grad_weight.view(w.shape)

            else:

                grad_weight = raw_grad_w



        return grad_input, grad_weight, None





# ==========================================

#  2. THE MODULAR LAYER (Clean Abstraction)

# ==========================================



class Narcissist(nn.Module):

    """

    Self-Excitatory Layer.

    

    Amplifies features that are statistically significant (far from the mean).

    Equivalent to a learnable, non-linear high-pass filter.

    """

    def __init__(self, num_features=None, dim=None):

        super().__init__()

        self.dim = dim

        self.num_features = num_features

        

        if num_features:

            self.weight = nn.Parameter(torch.empty(num_features))

        else:

            self.weight = nn.Parameter(torch.empty(1))

            

        self.reset_parameters()



    def reset_parameters(self):

        # Initialize close to 0 to start as a near-identity mapping (gate ≈ 0.5)

        # or slightly positive to encourage initial differentiation.

        nn.init.normal_(self.weight, mean=1.0, std=0.1)



    def _align_weight(self, x):

        """Dynamically reshapes weight to align with input channels."""

        if self.num_features is None:

            return self.weight

            

        # Create a shape tuple like (1, C, 1, 1) automatically

        view_shape = [1] * x.ndim

        

        # Heuristic: Assume channels are at dim 1 if dim 0 is batch

        channel_dim = 1 if x.ndim > 1 else 0

        

        # Safety check

        if x.shape[channel_dim] != self.num_features:

             # Fallback logic: Try to find the dimension that matches

             if self.num_features in x.shape:

                 channel_dim = x.shape.index(self.num_features)

             else:

                 raise ValueError(f"Input shape {x.shape} does not match num_features={self.num_features}")

                 

        view_shape[channel_dim] = self.num_features

        return self.weight.view(*view_shape)



    def forward(self, x):

        w = self._align_weight(x)

        return NarcissistFunction.apply(x, w, self.dim)



    def extra_repr(self):

        return f'num_features={self.num_features}, dim={self.dim}'



# ==========================================

#  3. TEST SUITE

# ==========================================

if __name__ == "__main__":

    from torch.autograd import gradcheck

    

    print("\n🔬 --- 1. Running Gradient Check ---")

    

    # Setup for 4D input (Batch, Channel, H, W)

    # We test with Double precision for numerical stability during gradcheck

    x = torch.randn(2, 2, 4, 4, dtype=torch.double, requires_grad=True)

    w = torch.randn(1, 2, 1, 1, dtype=torch.double, requires_grad=True)

    dims = (2, 3) 



    # Verify the custom backward pass against numerical approximation

    try:

        ok = gradcheck(NarcissistFunction.apply, (x, w, dims), eps=1e-6, atol=1e-4)

        print("✅ Gradient Check PASSED: Analytical implementation matches Numerical.")

    except Exception as e:

        print(f"❌ Gradient Check FAILED: {e}")



    print("\n🧠 --- 2. Module Integration Test ---")

    

    # Instantiate

    model = Narcissist(num_features=16, dim=(2,3)) # Spatial attention

    dummy_input = torch.randn(8, 16, 32, 32)

    

    # Forward

    y = model(dummy_input)

    print(f"Input:  {dummy_input.shape}")

    print(f"Output: {y.shape}")

    

    # Backward

    loss = y.mean()

    loss.backward()

    

    print(f"Weight Grad Norm: {model.weight.grad.norm().item():.6f}")

    print("✅ Full cycle executed successfully.")
