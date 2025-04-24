# reverse.py

import torch

class DiffusionReverse:
    """
    Implements the reverse diffusion process as described in the DDPM paper.
    Given a trained noise prediction model, this class iteratively denoises an image,
    starting from pure Gaussian noise.
    """
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02, device=None):
        """
        Initializes the reverse diffusion process w/ params
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.num_steps = num_steps
        
        # Create a linear beta schedule and compute related quantities
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # alphas_cumprod_prev: for t=0, it is 1, then shift right
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Compute the posterior variance for each timestep:
        # posterior_variance[t] = beta_t * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def p_sample(self, x_t, t):
        """
        Given the current noisy image x_t at timestep t, compute x_{t-1} using model
        returns denoised image tensor at timestep t-1.
        """

        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device)
        
        # (x_t, t) input for model
        predicted_noise = self.model(x_t, t_tensor)
        
        # extract scalar vals
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # reverse posterior mean
        # mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alphas_cumprod[t]) * predicted_noise)
        model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
        
        # for t > 0, add noise according to the posterior variance; for t == 0, return the mean.
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            x_prev = model_mean + torch.sqrt(variance) * noise
        else:
            x_prev = model_mean
        
        return x_prev
    
    def sample(self, shape):
        """
        Generates a sample image by starting from pure Gaussian noise and iteratively applying the reverse process.
        returns tensor approx of x_0
        """

        # start gaussian
        x_t = torch.randn(shape, device=self.device)
        
        # denoise until 0
        for t in reversed(range(self.num_steps)):
            x_t = self.p_sample(x_t, t)

            if t % 50 == 0:
                print(f"Sampling at timestep {t}")
        
        return x_t

