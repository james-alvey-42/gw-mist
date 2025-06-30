import torch
from typing import Tuple


class OnlineStandardizingLayer(torch.nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        stable: bool = False,
        epsilon: float = 1e-10,
        use_average_std: bool = False,
    ) -> None:
        """Accumulate mean and variance online using the "parallel algorithm" algorithm from [1].

        Args:
            shape: shape of mean, variance, and std array. do not include batch dimension!
            stable: (optional) compute using the stable version of the algorithm [1]
            epsilon: (optional) added to the computation of the standard deviation for numerical stability.
            use_average_std: (optional) ``True`` to normalize using std averaged over the whole observation, ``False`` to normalize using std of each component of the observation.

        References:
            [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        super().__init__()
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_M2", torch.zeros(shape))
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.shape = shape
        self.stable = stable
        self.use_average_std = use_average_std

    def _parallel_algorithm(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.shape[1:] == self.shape, "%s vs %s" % (x.shape[1:], self.shape)
        na = self.n.clone()
        nb = x.shape[0]
        nab = na + nb

        xa = self._mean.clone()
        xb = x.mean(dim=0)
        delta = xb - xa
        if self.stable:
            xab = (na * xa + nb * xb) / nab
        else:
            xab = xa + delta * nb / nab

        m2a = self._M2.clone()
        m2b = (
            x.var(dim=(0,), unbiased=False) * nb
        )  # do not use bessel's correction then multiply by total number of items in batch.
        m2ab = m2a + m2b + delta ** 2 * na * nb / nab
        return nab, xab, m2ab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.n, self._mean, self._M2 = self._parallel_algorithm(x)
        return (x - self.mean) / self.std

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def var(self) -> torch.Tensor:
        if self.n > 1:
            return self._M2 / (self.n - 1)
        else:
            return torch.zeros_like(self._M2)

    @property
    def std(self) -> torch.Tensor:
        if self.use_average_std:
            return torch.sqrt(self.var + self.epsilon).mean()
        else:
            return torch.sqrt(self.var + self.epsilon)
