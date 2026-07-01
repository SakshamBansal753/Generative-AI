import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self,timesteps=1000,beta_start=1e-4,beta_end=0.02):
        self.timesteps=timesteps
        self.betas=torch.linspace(beta_start,beta_end,timesteps)
        self.alphas=1.0-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,dim=0)
        self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)

        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-self.alphas_cumprod)

        self.posterior_variance=self.betas*(1.0-self.alphas_cumprod_prev)/ (1.0-self.alphas_cumprod)

    @staticmethod

    def _extract(a,t,x_shape):
        batch_size=t.shape[0]
        out=a.gather(-1,t)
        return out.reshape(batch_size, *((1,)*(len(x_shape)-1)))
    
    def q_sample(self,x0,t,noise=None):
        if noise is None:
            noise=torch.randn_like(x0)
        sqrt_alpha_bar=self._extract(self.sqrt_alphas_cumprod,t,x0.shape)
        sqrt_one_minus_alpha_bar=self._extract(self.sqrt_one_minus_alphas_cumprod,t,x0.shape)
        return sqrt_alpha_bar*x0+sqrt_one_minus_alpha_bar*noise,noise

    def training_loss(self,model,x0,t,y=None):
        x_noisy,noise=self.q_sample(x0,t)
        predicted_noise=model(x_noisy,t,y) if y is not None else model(x_noisy,t)
        return F.mse_loss(predicted_noise,noise)
    @torch.no_grad()
    def p_sample(self,model,x_t,t,y=None):
        t_batch=torch.full((x_t.shape[0],),t,dtype=torch.long,device=x_t.device)
        predicted_noise=model(x_t,t_batch,y) if y is not None else model(x_t,t_batch)
        beta_t=self.betas[t]
        alpha_t=self.alphas[t]
        alpha_bar_t=self.alphas_cumprod[t]
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * predicted_noise)

        if t==0:
            return mean
        else:
            noise=torch.randn_like(x_t)
            var=self.posterior_variance[t]
            return mean+torch.sqrt(var)*noise
    @torch.no_grad()
    def sample(self,model,shape,y=None,device="cpu"):
        model.eval()
        x_t=torch.randn(shape,device=device)
        for t in reversed(range(self.timesteps)):
            x_t=self.p_sample(model,x_t,t,y)
        model.train()
        return x_t