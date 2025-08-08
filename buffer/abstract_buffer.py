from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from gymnasium.core import ObsType, SupportsFloat

# state, action, reward, next_state, terminated, truncated, info
Transition = Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]


class AbstractBuffer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """Add transition to buffer.

        Parameters
        ----------
        state : np.ndarray
            State
        action : int | float
            Action
        reward : float
            Reward
        next_state : np.ndarray
            Next state
        done : bool
            Done (terminated or truncated)
        info : dict
            Info dict
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args: tuple, **kwargs: dict) -> Iterable[Transition]:
        """Sample from buffer.

        Returns
        -------
        Iterable[Transition]
            Iterable (e.g. list) of transitions
        """
        raise NotImplementedError
