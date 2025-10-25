import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
import hydra

class Actor(nn.Module):
    def __init__(self, cfg, act_dim: int):
        super().__init__()
        self.sequential = hydra.utils.instantiate(cfg)
        self.mu = nn.LazyLinear(act_dim)
        self.log_std = nn.LazyLinear(act_dim)
        self.device = 'cpu'
    
    def forward(self, obs):
        h = self.sequential(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        dist = SquashedDiagGaussianDistribution(mu, log_std.exp())
        return dist
    
    def predict(self, obs, deterministic=True):
        obs = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self(obs)
        act = dist.mean if deterministic else dist.sample()
        return act.cpu().numpy().squeeze(0)
    
    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.device = device
        return self

class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q1 = hydra.utils.instantiate(cfg).append(nn.LazyLinear(1))
        self.q2 = hydra.utils.instantiate(cfg).append(nn.LazyLinear(1))

    def forward(self, obs, act):
        obs_act = torch.cat([obs, act], dim=1)
        return self.q1(obs_act).squeeze(1), self.q2(obs_act).squeeze(1)

class SquashedDiagGaussianDistribution:
    def __init__(self, mu, std, epsilon=1e-6):
        self.distribution = Normal(mu, std)
        self.epsilon = epsilon
    
    @property
    def mean(self):
        return torch.tanh(self.distribution.mean)
    
    def sample(self):
        return torch.tanh(self.distribution.sample())
    
    def log_prob(self, act, gaussian_act=None):
        if gaussian_act is None:
            eps = torch.finfo(act.dtype).eps
            clamped_act = act.clamp(min=-1.0 + eps, max=1.0 - eps)
            gaussian_act = 0.5 * (clamped_act.log1p() - (-clamped_act).log1p())
        log_prob = self.distribution.log_prob(gaussian_act).sum(dim=1)
        log_prob -= torch.sum(torch.log(1 - act**2 + self.epsilon), dim=1)
        return log_prob

    def act_log_prob(self):
        gaussian_act = self.distribution.rsample()
        act = torch.tanh(gaussian_act)
        log_prob = self.log_prob(act, gaussian_act)
        return act, log_prob

class FE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.convnet = hydra.utils.instantiate(cfg).append(nn.Flatten())
    
    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h

class FixedStdActor(nn.Module):
    def __init__(self, cfg, act_dim):
        super().__init__()
        self.mu = hydra.utils.instantiate(cfg).append(nn.LazyLinear(act_dim))
    
    def forward(self, obs, std):
        mu = self.mu(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist

class TrunkHeadCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trunk = hydra.utils.instantiate(cfg.trunk)
        self.q1 = hydra.utils.instantiate(cfg.head).append(nn.LazyLinear(1))
        self.q2 = hydra.utils.instantiate(cfg.head).append(nn.LazyLinear(1))

    def forward(self, obs, act):
        h = self.trunk(obs)
        h_act = torch.cat([h, act], dim=1)
        return self.q1(h_act).squeeze(1), self.q2(h_act).squeeze(1)

class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)