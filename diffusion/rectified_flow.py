import torch as th
import numpy as np
from typing import Any, Dict, List, Optional

class RectifiedFlow:
    """
    Rectified Flow implementation for straight-line flow from noise to data.
    
    Key differences from diffusion:
    - Uses linear interpolation: x_t = (1-t)*x_0 + t*x_1 where x_0~N(0,I), x_1 is data
    - Model predicts velocity field: v_t = x_1 - x_0
    - No noise schedule - just uniform time sampling
    """
    
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        
    def sample_time(self, batch_size: int, device: th.device) -> th.Tensor:
        """Sample random timesteps uniformly from [0, 1]"""
        return th.rand(batch_size, device=device)
    
    def interpolate(self, x_0: th.Tensor, x_1: th.Tensor, t: th.Tensor) -> th.Tensor:
        """
        Linear interpolation between noise x_0 and data x_1.
        x_t = (1-t)*x_0 + t*x_1
        
        Args:
            x_0: noise tensor [B, C, H, W]  
            x_1: data tensor [B, C, H, W]
            t: time tensor [B] with values in [0, 1]
        """
        t = t.view(-1, 1, 1, 1)  # Reshape for broadcasting
        return (1 - t) * x_0 + t * x_1
    
    def compute_velocity(self, x_0: th.Tensor, x_1: th.Tensor) -> th.Tensor:
        """
        Compute the true velocity field: v = x_1 - x_0
        
        Args:
            x_0: noise tensor [B, C, H, W]
            x_1: data tensor [B, C, H, W]  
        """
        return x_1 - x_0
    
    def training_losses(self, model, x_start: th.Tensor, t: Optional[th.Tensor] = None, 
                       model_kwargs: Optional[Dict[str, Any]] = None, 
                       noise: Optional[th.Tensor] = None) -> Dict[str, th.Tensor]:
        """
        Compute rectified flow training loss.
        
        Args:
            model: velocity prediction model
            x_start: clean data tensor [B, C, H, W]
            t: time tensor [B] (if None, will be sampled)
            model_kwargs: conditioning arguments for model
            noise: noise tensor [B, C, H, W] (if None, will be sampled)
            
        Returns:
            Dictionary with 'loss' key containing MSE loss
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        if noise is None:
            noise = th.randn_like(x_start)
            
        if t is None:
            t = self.sample_time(x_start.shape[0], x_start.device)
        
        # Convert t to discrete timesteps for model (if model expects discrete steps)
        t_discrete = (t * (self.num_timesteps - 1)).long()
        
        # Linear interpolation: x_t = (1-t)*noise + t*data
        x_t = self.interpolate(noise, x_start, t)
        
        # True velocity field: v = data - noise  
        true_velocity = self.compute_velocity(noise, x_start)
        
        # Model prediction
        model_output = model(x_t, t_discrete, **model_kwargs)
        # Handle learn_sigma case - split output if it's double the input channels
        if model_output.shape[1] == 2 * x_start.shape[1]:
            predicted_velocity, _ = th.split(model_output, x_start.shape[1], dim=1)
        else:
            predicted_velocity = model_output
        
        # MSE loss between predicted and true velocity
        mse_loss = th.nn.functional.mse_loss(predicted_velocity, true_velocity, reduction='none')
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))  # Mean over all dims except batch
        
        return {
            'loss': mse_loss,
            'mse': mse_loss  # For compatibility with diffusion interface
        }

    def sample_step(self, model, x_t: th.Tensor, t: th.Tensor, dt: float,
                   model_kwargs: Optional[Dict[str, Any]] = None) -> th.Tensor:
        """
        Single Euler step for sampling: x_{t+dt} = x_t + dt * v_theta(x_t, t)
        
        Args:
            model: velocity prediction model
            x_t: current state [B, C, H, W]
            t: current time [B]
            dt: step size
            model_kwargs: conditioning arguments
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Convert continuous time to discrete timesteps  
        t_discrete = (t * (self.num_timesteps - 1)).long()
        
        # Predict velocity
        with th.no_grad():
            velocity = model(x_t, t_discrete, **model_kwargs)
        
        # Euler step
        x_next = x_t + dt * velocity
        return x_next
    
    def dopri5_step(self, model, x_t: th.Tensor, t: th.Tensor, dt: float,
                    model_kwargs: Optional[Dict[str, Any]] = None, 
                    atol: float = 1e-5, rtol: float = 1e-3) -> tuple:
        """
        Adaptive DOPRI5 (Dormand-Prince) step with error control
        
        Returns:
            x_next: next state
            dt_next: suggested next step size
            error: estimated error
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # DOPRI5 Butcher tableau coefficients
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        
        b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        b_hat = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Store k values
        k = []
        
        with th.no_grad():
            # k1
            t_discrete = (t * (self.num_timesteps - 1)).long()
            k1 = model(x_t, t_discrete, **model_kwargs)
            k.append(k1)
            
            # k2 through k7
            for i in range(1, 7):
                t_stage = t + c[i] * dt
                t_stage_discrete = (t_stage * (self.num_timesteps - 1)).long()
                
                x_stage = x_t.clone()
                for j in range(i):
                    x_stage = x_stage + dt * a[i][j] * k[j]
                
                k_i = model(x_stage, t_stage_discrete, **model_kwargs)
                k.append(k_i)
            
            # 5th order solution
            x_next = x_t.clone()
            for i in range(7):
                x_next = x_next + dt * b[i] * k[i]
            
            # 4th order solution for error estimation
            x_hat = x_t.clone()
            for i in range(7):
                x_hat = x_hat + dt * b_hat[i] * k[i]
            
            # Error estimation
            error = th.abs(x_next - x_hat)
            error_norm = th.sqrt(th.mean(error ** 2, dim=list(range(1, len(error.shape)))))
            
            # Adaptive step size control
            tolerance = atol + rtol * th.maximum(th.abs(x_t), th.abs(x_next)).mean(dim=list(range(1, len(x_t.shape))))
            
            # Safety factor and step size adaptation
            safety = 0.9
            error_ratio = error_norm / tolerance
            
            # Prevent division by zero
            error_ratio = th.clamp(error_ratio, min=1e-10)
            
            # New step size calculation
            dt_factor = safety * th.pow(error_ratio, -1/5)
            dt_factor = th.clamp(dt_factor, min=0.2, max=5.0)
            dt_next = dt * dt_factor.mean().item()
            
        return x_next, dt_next, error_norm.mean().item()
    
    def sample_dopri5(self, model, shape: tuple, num_steps: int = 20,
                      model_kwargs: Optional[Dict[str, Any]] = None,
                      device: Optional[th.device] = None,
                      atol: float = 1e-5, rtol: float = 1e-3) -> th.Tensor:
        """
        Generate samples using adaptive DOPRI5 integration
        
        Args:
            model: velocity prediction model
            shape: output shape [B, C, H, W]
            num_steps: target number of integration steps (adaptive)
            model_kwargs: conditioning arguments
            device: device for computation
            atol: absolute tolerance for error control
            rtol: relative tolerance for error control
        """
        if device is None:
            device = next(model.parameters()).device
            
        if model_kwargs is None:
            model_kwargs = {}
        
        # Start from noise
        x = th.randn(*shape, device=device)
        
        # Initial step size
        dt = 1.0 / num_steps
        t = 0.0
        
        steps_taken = 0
        max_steps = num_steps * 3  # Safety limit
        
        while t < 1.0 and steps_taken < max_steps:
            # Ensure we don't overshoot
            if t + dt > 1.0:
                dt = 1.0 - t
            
            t_tensor = th.full((shape[0],), t, device=device)
            
            # Take DOPRI5 step
            x_next, dt_next, error = self.dopri5_step(
                model, x, t_tensor, dt, model_kwargs, atol, rtol
            )
            
            # Accept or reject step based on error
            if error < 1.0:  # Accept step
                x = x_next
                t += dt
                steps_taken += 1
                
            # Update step size for next iteration
            dt = min(dt_next, 1.0 - t)
            
            # Minimum step size to prevent infinite loops
            dt = max(dt, 1e-6)
            
        return x
    
    def sample_loop(self, model, shape: tuple, num_steps: int = 50,
                   model_kwargs: Optional[Dict[str, Any]] = None,
                   device: Optional[th.device] = None,
                   progress: bool = False) -> th.Tensor:
        """Sample with optional progress bar (for compatibility with diffusion interface)"""
        if progress:
            try:
                from tqdm.auto import tqdm
                steps = tqdm(range(num_steps))
            except ImportError:
                steps = range(num_steps)
        else:
            steps = range(num_steps)
            
        if device is None:
            device = next(model.parameters()).device
            
        if model_kwargs is None:
            model_kwargs = {}
            
        x = th.randn(*shape, device=device) 
        dt = 1.0 / num_steps
        
        for i in steps:
            t = th.full((shape[0],), i * dt, device=device)
            x = self.sample_step(model, x, t, dt, model_kwargs)
            
        return x

def create_rectified_flow(num_timesteps: int = 1000) -> RectifiedFlow:
    """Factory function to create RectifiedFlow (matches diffusion create_diffusion interface)"""
    return RectifiedFlow(num_timesteps=num_timesteps)
