import torch
import math

class DiffusionForward:
    """
    Implements the forward diffusion process as described in the DDPM paper.
    This class sets up the noise scheduler and provides a method to sample the noisy image x_t from the original image x_0.
    """

    def __init__(self, num_steps=1000, beta_start=1e-5, beta_end=0.02, device=None):
        """
        Initializes the diffusion forward process scheduler.
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.num_steps = num_steps
    
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device) # Linear Scheduler
        # self.betas = self.cosine_beta_schedule(num_steps).to(self.device) # Cosine Scheduler <- DOESN'T WORK VERY WELL
        
        # Alphas for closed form sampling 
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute square roots 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        DON'T USE DOESN'T WORK VERY WELL
        Cosine schedule as proposed in 
        "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
        
        s = offset to prevent undef behavior at 0 (0.008 default from DDPM paper)
                    
        return 1D tensor of betas 
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)  # [0, 1, 2, ..., T]
        
        # alphas_cumprod = (cos( (t/T + s) / (1+s) * pi/2 ))^2
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        # Normalize so that alphas_cumprod[0] = 1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # betas[i] = 1 - (alpha_{i+1} / alpha_i)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clamp betas to prevent numerical issues
        betas = torch.clip(betas, 0, 0.999)
        return betas

    def q_sample(self, x0, t, noise=None):
        """
        samples a noisy image x_t from the original image x0 at a given timestep t.
        x_t = sqrt(alphas_cumprod[t]) * x0 + sqrt(1 - alphas_cumprod[t]) * noise
        
        x0 = orig tensor (batch, channel, H, W)

        returns noisy image tensor x_t
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Ensure t is a tensor of shape (batch_size,)
        if isinstance(t, int):
            t = torch.tensor([t] * x0.shape[0], device=self.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x0.shape[0])
        
        # Get the appropriate cumulative products for each timestep and reshape for broadcasting
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Return the noisy image using the closed-form expression
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise