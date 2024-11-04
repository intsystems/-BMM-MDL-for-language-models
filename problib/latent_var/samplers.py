from typing import List, Optional
import torch
import torch.nn as nn
import torch_struct


class BaseSampler(nn.Module):
    def __init__(self, D: int, K: int):
        super().__init__()
        self.D = D  
        self.K = K  

    def sample(self, N: int) -> torch.Tensor:
        raise NotImplementedError("Each sampler must implement the sample method.")

    def get_top_k_features(self) -> List[int]:
        raise NotImplementedError("Each sampler must implement the get_top_k_features method.")


class ConditionalPoissonSampler(BaseSampler):
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.weights = nn.Parameter(torch.rand(D)) 

    def get_device(self) -> torch.device:
        return self.weights.device

    def run_on_semiring(self, sr, cache: Optional[List[torch.Tensor]]) -> torch.Tensor:
        weights_sr = sr.convert(self.weights)  
        return sr.unconvert(self._compute_S_N_n(sr, weights_sr, self.D, self.K, cache))

    def entropy(self, cache: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return self.run_on_semiring(torch_struct.EntropySemiring, cache if cache is not None else [])

    def logZ(self, cache: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return self.run_on_semiring(torch_struct.LogSemiring, cache if cache is not None else [])

    def logprob(self, mask: torch.Tensor, logZ: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logZ is None:
            logZ = self.logZ()

        return mask.float() @ self.weights - logZ

    def sample(self, num_samples: int, log_cache: List[torch.Tensor]) -> torch.Tensor:
        num_to_sample = self.K * torch.ones(num_samples, device=self.get_device(), dtype=torch.long)
        return self._sample(num_samples, num_to_sample, log_cache)

    def _sample(self, num_samples: int, num_to_sample: torch.Tensor, log_cache: List[torch.Tensor]) -> torch.Tensor:
        samples = torch.zeros(num_samples, self.D, device=self.weights.device, dtype=torch.long)
        for d in reversed(range(1, self.D + 1)):
            num_sampled_dimensions = samples.sum(dim=1)
            num_dimensions_to_sample = num_to_sample - num_sampled_dimensions

            sample_dim_prob = (
                self.weights[d - 1]
                + self._get_from_cache(d - 1, num_dimensions_to_sample - 1, log_cache)
                - self._get_from_cache(d, num_dimensions_to_sample, log_cache)
            ).exp()

            sample_dim_prob.masked_fill_(num_dimensions_to_sample >= d, 1.0)
            sample_dim_prob.masked_fill_(num_sampled_dimensions >= num_to_sample, 0.0)

            samples[:, d - 1] = torch.bernoulli(sample_dim_prob)

        assert (samples.sum(dim=1) == num_to_sample).all()
        return samples

    def _get_from_cache(self, N: int, n: int, cache: List[torch.Tensor]) -> torch.Tensor:
        return cache[N - 1][..., n]

    def _compute_S_N_n(self, sr, weights: torch.Tensor, N: int, n: int, cache: List[torch.Tensor]) -> torch.Tensor:
        assert len(cache) == 0, "Cache should be empty when starting the calculation."

        one = sr.convert(torch.tensor(0.0, device=self.get_device()))
        sr.one_(one)
        zero = sr.convert(torch.tensor(0.0, device=self.get_device()))
        sr.zero_(zero)
        zero_t = sr.convert(torch.zeros(1, device=self.get_device()))
        sr.zero_(zero_t)

        start = sr.convert(torch.zeros(N + 1, device=self.get_device()))
        sr.zero_(start)
        start[..., 1 - 1] = one
        start[..., 2 - 1] = weights[..., 0]
        cache.append(start)

        for d in range(1, N):
            S_less_curr = cache[d - 1]
            S_less_less = torch.cat([zero_t, S_less_curr[..., :-1]], dim=-1)

            S_curr = sr.plus(sr.mul(weights[..., d], S_less_less), S_less_curr)
            cache.append(S_curr)

        assert len(cache) == N
        return cache[N - 1][..., n]

    def get_top_k_features(self) -> List[int]:
        _, top_k_indices = torch.topk(self.weights, self.K)
        return top_k_indices.tolist()


class PoissonSampler(BaseSampler):
    def __init__(self, D: int, K: int):
        super().__init__(D, K)
        self.weights = nn.Parameter(torch.rand(D))

    def marginals(self) -> torch.Tensor:
        w = self.weights.exp()
        return w / (1 + w)
    
    def log_Z(self) -> torch.Tensor:
        w = self.weights.exp()
        return torch.sum(torch.log1p(w))
    
    def entropy(self) -> torch.Tensor:
        w = self.weights.exp()
        logs = -torch.log(w)
        return torch.dot(self.marginals(), logs) + self.log_Z()
    
    def logprob(self, mask: torch.Tensor) -> torch.Tensor:
        w = self.weights.exp()
        logs = torch.log(w)
        return mask.float() @ logs - self.log_Z()

    def sample(self, N: int) -> torch.Tensor:
        return torch.bernoulli(self.marginals().expand(N, -1)).bool()

    def get_top_k_features(self) -> List[int]:
        _, top_k_indices = torch.topk(self.marginals(), self.K)
        return top_k_indices.tolist()

def get_sampler(sampler_type: str, D: int, K: int) -> BaseSampler:
    if sampler_type == "conditional_poisson":
        return ConditionalPoissonSampler(D, K)
    elif sampler_type == "poisson":
        return PoissonSampler(D, K)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
