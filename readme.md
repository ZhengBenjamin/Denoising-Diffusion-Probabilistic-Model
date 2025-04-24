# Probabilistic Perspective on Diffusion Models: Theory and Implementation  
**CSDS 491 Probabilistic Graphical Models**  
Case Western Reserve University
Spring 2025    
Benjamin Zheng

View the final report in **Final_Report.pdf** in the written folder.

## Outcomes

#### Human64 Model

The human64 model was trained on 64x64 images from Flickr-Faces-HQ dataset. t = 750 steps, 386 base channels, linear beta schedule (between 1e-4, 0,02):

**Final trained samples:**
![human64ext](https://github.com/user-attachments/assets/8bb5faa7-4834-48b3-bc8a-1329b547b65f)
*human64 model: t = 750, base_channels = 384, time_emb_dim = 512, batch_size = 16, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

**Reverse Process:**

![human64rev](https://github.com/user-attachments/assets/c9e3d97d-d508-4d12-a24c-7f96891a6a0d)
*human64 reverse process: 75 step transitions, t = 750, base_channels = 512, time_emb_dim = 512, batch_size = 16, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

**Progression figures:**

![human5k](https://github.com/user-attachments/assets/e59ff0f2-5201-41cd-9bf1-5c7ba3e5f340)
![human10k](https://github.com/user-attachments/assets/acce4044-b74a-4e5c-83bb-6e2895cfb37f)
![human20k](https://github.com/user-attachments/assets/c69a2bb9-e869-4063-b932-546aeae71aef)
![human90k](https://github.com/user-attachments/assets/588f73a2-c495-4938-a597-001ad0647cbf)

Although not perfect, the model generated varied human faces. With more memory and training time, the results could be further improved.

#### DrunkSimp64 Model

The drunksimp64 model was trained on 64x64 images with t = 500, 256 base channels, linear beta schedule (1e-4, 0.02)  

**Final trained samples:**

![simp64ext](https://github.com/user-attachments/assets/619ceca3-98f3-4164-bcf6-ea47c1ab7be3)
*drunkSimpson64 model: t = 500, base_channels = 512, time_emb_dim = 512, batch_size = 16, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

**Reverse Process:**

![simp64rev](https://github.com/user-attachments/assets/33115b06-ea04-411e-bd6d-bce5988e203a)
*drunkSimpson64 reverse process: 75 step transitions, t = 750, base_channels = 512, time_emb_dim = 512, batch_size = 16, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

#### Dog32 Model

The dog32 model was trained on 32x32 images with t = 1000, 256 base channels, linear beta schedule (1e-4, 0.02)  

**Final trained samples:**

![dog32ext](https://github.com/user-attachments/assets/00a2c9ac-4f4c-4ca0-889d-3eb824c0da67)
*dog32 model: t = 500, base_channels = 256, time_emb_dim = 512, batch_size = 32, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

**Reverse Process:**

![dog32rev](https://github.com/user-attachments/assets/f4d1f67f-aa44-47e9-a2ca-9f6adaa9d5b5)
*dog32 reverse process: 100 step transitions, t = 1000, base_channels = 256, time_emb_dim = 512, batch_size = 32, lr = 1e-4, linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.02%29)

---

### Diffusion Models

Diffusion models are a class of generative models that learn to generate data by reversing a diffusion process.  
The diffusion process gradually adds noise to the data, and the model learns to reverse this process, generating new samples from random noise.  
The key idea is to model the data distribution as a Markov chain, where each step in the chain corresponds to a diffusion step.  
The forward process involves adding Gaussian noise to the data, while the reverse process learns to denoise the data step by step.  
The model is trained using a variational approach, optimizing a lower bound on the data likelihood.

---

## Theoretical Background

### Forward Process

The forward process in diffusion models involves gradually adding noise to the data.  
This is typically done using a Gaussian noise process, where at each step, a small amount of noise is added to the data.  
The forward process can be described mathematically as follows:  

![](https://latex.codecogs.com/png.latex?q%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%3D%5Cprod_%7Bt%3D1%7D%5ETq%5Cleft%28x_t%5Cmid%20x_%7Bt-1%7D%5Cright%29)

Where **x₀** is the original data, **xₜ** is the noisy data at time step *t*, and  
![](https://latex.codecogs.com/png.latex?q%5Cleft%28x_t%5Cmid%20x_0%5Cright%29%3DN%5Cleft%28%5Csqrt%7B%5Coverline%7B%5Calpha_t%7D%7D%20x_0%2C1-%5Coverline%7B%5Calpha_t%7D%5Cright%29)  

The distribution above is parameterized by:  

![](https://latex.codecogs.com/png.latex?%5Calpha_t%3D1-%5Cbeta_t%2C%5Cquad%20%5Coverline%7B%5Calpha_t%7D%3D%5Cprod_%7Bs%3D1%7D%5Et%5Calpha_s)

This implementation is only possible because we are using a fixed scheduled $\beta_t$ (More on this in the next section).  
Using this, any time step could be determined without needing to go through the entire chain.

#### Figure: Forward Process

![forward2](https://github.com/user-attachments/assets/c98b0209-4d6c-43b8-a3d0-8a20ac65c158)
*Forward process of human64 model from t₀ to t₁₀₀ with linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%281e%5E%7B-4%7D%2C%200.05%29)

---

### Scheduler

The variance scheduler ![](https://latex.codecogs.com/png.latex?%5Cbeta_t) controls the amount of noise added at each step in the forward process.  
There are many different ways to implement this scheduler, but the most common approach is to use a linear or exponential schedule.  
The linear schedule increases the variance linearly over time, starting from a small value and gradually increasing to a larger value.  
Typically, it would look something like this:  

![](https://latex.codecogs.com/png.latex?%5Cbeta_t%3D%5Cbeta_0%2B%5Cfrac%7Bt-1%7D%7BT-1%7D%28%5Cbeta_T-%5Cbeta_0%29)

Where ![](https://latex.codecogs.com/png.latex?%5Cbeta_0) is the initial variance, ![](https://latex.codecogs.com/png.latex?%5Cbeta_T) is the final variance, and **T** is the total number of steps.  
Having a larger *t* allows for a smaller ![](https://latex.codecogs.com/png.latex?%5Cbeta) (step size) while still approximating the same limiting distribution.  
Smaller time steps allow for less ambiguity in determining previous states.  
Generally, the smaller the ![](https://latex.codecogs.com/png.latex?%5Cbeta), the smaller the variance.  
The true reverse process will have the same functional form of the forward process using an infinitely small step size ![](https://latex.codecogs.com/png.latex?%5Cbeta).

---

### Reverse Process

The reverse process in diffusion models involves learning to denoise the data step by step.  
The process is similar to the forward process, but instead of adding noise, we learn to remove it.  
Mathematically, we can describe the reverse process as follows:  

![](https://latex.codecogs.com/png.latex?p_%5Ctheta%5Cleft%28x_%7B0%3AT%7D%5Cright%29%3Dp%5Cleft%28x_T%5Cright%29%5Cprod_%7Bt%3D1%7D%5ETp_%5Ctheta%5Cleft%28x_%7Bt-1%7D%5Cmid%20x_t%5Cright%29)

Where ![](https://latex.codecogs.com/png.latex?p_%5Ctheta%5Cleft%28x_%7Bt-1%7D%5Cmid%20x_t%5Cright%29) is the learned transition distribution from **xₜ** to **xₜ₋₁**.  
Here, ![](https://latex.codecogs.com/png.latex?p%5Cleft%28x_T%5Cright%29) is the pure noise distribution from our forward process (Gaussian) and the product represents the chain of conditionals.  
![](https://latex.codecogs.com/png.latex?%5Ctheta) represents the trainable parameters (weights and bias) of the model.

We can break this down further into:  

![](https://latex.codecogs.com/png.latex?p_%5Ctheta%5Cleft%28x_%7Bt-1%7D%5Cmid%20x_t%5Cright%29%3DN%5Cleft%28%5Cmu_%5Ctheta%5Cleft%28x_t%2Ct%5Cright%29%2C%5Csum_%5Ctheta%5Cleft%28x_t%2Ct%5Cright%29%5Cright%29)

which takes time **t** and our sample **xₜ**.  
Since the variance scheduler is not constant, input *t* allows us to account for the forward process variance scheduler.  
At each time step, different noise levels appear, and the model must learn how to undo these steps individually.  
During inference, we start from absolute Gaussian noise and sample from the learned reverse transitions until reaching **x₀**.

#### Figure: Reverse Process

![reverse3](https://github.com/user-attachments/assets/e70c9a36-a369-4448-85e3-f7d79e40e310)
*Reverse process of human64 model from t₇₅₀ to t₀ with linear* ![](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Cin%20%280.0001%2C%200.05%29)

---

### Objective Function

The objective function for training diffusion models is based on the variational lower bound on the data likelihood.

Let **x₁, x₂, …, x_T** be the latent variables representing the noisy image at each step.  
Our observed value is **x₀**, the original image.  
The log-likelihood of the data can be written as:

![](https://latex.codecogs.com/png.latex?%5Clog%5Cleft%28p%5Cleft%28x_0%5Cright%29%5Cright%29%20%5Cge%20E_%7Bq%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%7D%5B%5Clog%5Cleft%28p_%5Ctheta%5Cleft%28x_0%5Cmid%20x_%7B1%3AT%7D%5Cright%29%5Cright%29%5D-%20D_%7BKL%7D%5Cleft%28q%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%5Cparallel%20p%5Cleft%28x_%7B1%3AT%7D%5Cright%29%5Cright%29)

Here, the first term is the reconstruction term measuring how well the model recovers **x₀** from the noisy observations, and the second term (the KL divergence) regularizes the latent distribution against the known prior.

This expression can be factored into contributions from each individual step:

![](https://latex.codecogs.com/png.latex?E_q%5B%5Clog%5Cleft%28p_%5Ctheta%5Cleft%28x_0%5Cmid%20x_%7B1%3AT%7D%5Cright%29%5Cright%29%5D-E_q%5B%5Clog%5Cfrac%7Bq%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%7D%7Bp_%5Ctheta%5Cleft%28x_%7B1%3AT%7D%5Cright%29%7D%5D)

![](https://latex.codecogs.com/png.latex?E_q%5B%5Clog%5Cleft%28p_%5Ctheta%5Cleft%28x_0%5Cmid%20x_%7B1%3AT%7D%5Cright%29%5Cright%29&plus;%5Clog%5Cfrac%7Bp_%5Ctheta%5Cleft%28x_%7B1%3AT%7D%5Cright%29%7D%7Bq%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%7D%5D)

![](https://latex.codecogs.com/png.latex?E_q%5B%5Clog%5Cfrac%7Bp_%5Ctheta%5Cleft%28x_%7B1%3AT%7D%5Cright%29%7D%7Bq%5Cleft%28x_%7B1%3AT%7D%5Cmid%20x_0%5Cright%29%7D%5D)

![](https://latex.codecogs.com/png.latex?E_q%5B%5Clog%5Cleft%28p_%5Ctheta%5Cleft%28x_T%5Cright%29&plus;%5Csum_%7Bt%5Cge1%7D%5Clog%5Cfrac%7Bp_%5Ctheta%5Cleft%28x_%7Bt-1%7D%5Cmid%20x_t%5Cright%29%7D%7Bq%5Cleft%28x_t%5Cmid%20x_%7Bt-1%7D%5Cright%29%7D%5Cright%29%5D)

---

## Implementation

After establishing the mathematical foundation for both the forward and reverse processes, a UNet-based architecture is used to learn the denoising function.  
The reverse process begins with pure Gaussian noise and iteratively removes noise until an approximation of the original image is reached.  
This iterative denoising is powered by a learned model that predicts the noise at each time step.

### Dataset and Pre-processing

Two datasets were used to train the model: the *Cat vs Dog* dataset and the *Flickr-Faces-HQ* dataset from Kaggle.  
For this implementation, only dog images (500 samples) were used from the Cat vs Dog set. The images were downsampled to 32x32 and 64x64 pixels.  
The Flickr-Faces-HQ dataset provided around 52k images that were downsampled to 64x64 pixels and 10k images selected for training.  
Images were converted to numpy arrays with 3 channels (RGB) and transformed into tensors.  
Normalization and transformation to the [0, 1] range were performed on the GPU.

---

### UNet Architecture

A UNet-based architecture (adapted from [DDPM](https://doi.org/10.48550/arXiv.2006.11239)) is used to predict the noise.  
The architecture consists of:
- **Time Embedding:** Uses a sinusoidal function to encode the timestep which is then processed with a linear layer.
- **Downsampling:** Convolutional layers with leaky ReLU and batch normalization to reduce spatial resolution.
- **Upsampling:** Transpose convolutions to regain the original resolution, incorporating skip connections.
- **Residual Blocks:** To assist in gradient flow and preserve information between layers.

The model architecture and forward pass are defined in `model.py` (not shown here).

---

### Forward Process

The forward process uses a linear schedule for ![](https://latex.codecogs.com/png.latex?%5Cbeta_t) and precomputes cumulative products for ![](https://latex.codecogs.com/png.latex?%5Coverline%7B%5Calpha_t%7D) for efficiency.  
This allows sampling of any timestep in closed form and leverages GPU-accelerated vectorized operations.

---

### Reverse Process

In the reverse process, quantities such as the posterior variance for each timestep are precomputed.  
The noise prediction model is then used to compute the mean of the reverse transition:  

![](https://latex.codecogs.com/png.latex?%5Cmu_%5Ctheta%5Cleft%28x_t%2Ct%5Cright%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Calpha_t%7D%7D%5Cleft%28x_t-%5Cfrac%7B1-%5Calpha_t%7D%7B%5Csqrt%7B1-%5Coverline%7B%5Calpha_t%7D%7D%7D%5Cepsilon_%5Ctheta%5Cleft%28x_t%2Ct%5Cright%29%5Cright%29)

For timesteps t > 0, an appropriate amount of noise is added based on the posterior variance to maintain the stochastic nature of the process.

---

### Limitations and Modifications

Due to GPU memory constraints, adjustments were made including reducing base channels, time embedding dimensionality, image resolution, batch sizes, and number of timesteps.  
Training instability was countered by tuning hyperparameters such as the learning rate (set to ![](https://latex.codecogs.com/png.latex?1e%5E%7B-4%7D)), and the number of timesteps (typically between 750 and 1000).

Additional notes:  
This program is designed with CUDA in mind (Nvidia GPU required). Alternative backends (Apple MPS, ROCm) may be possible. Running on CPU will be considerably slower.

---

## Bibliography

- **Dog vs Cat:** AnthonyTherrien, yavuzibr. (2024, September 28). *Dog vs cat.* [Kaggle](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)
- **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep residual learning for image recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- **Denoising Diffusion Probabilistic Models:** Ho, J., Jain, A., & Abbeel, P. (2020). [arXiv:2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
- **Improved Denoising Diffusion Probabilistic Models:** Nichol, A., & Dhariwal, P. (2021). [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- **Flickr-Faces-HQ:** Rougetet, A. (2020). [Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)

---
