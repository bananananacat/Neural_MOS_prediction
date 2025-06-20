from typing import List

import numpy as np

import torch


class BaseModel:
    def __init__(self, weights: str) -> None:
        self.model = self._load_weights(weights)

    def _load_weights(self, weights: str) -> torch.nn.Module:
        """
        Load model weights from the specified path or huggingface path.

        Returns:
            A PyTorch model or Huggingface model with loaded weights
        """
        pass

    def predict(self, audios: np.ndarray) -> List[float]:
        audios = self.preprocess_audios(audios)
        return self.forward(audios)

    def preprocess_audios(self, audios: torch.Tensor) -> torch.Tensor:
        """
        Batched preprocessing
        """
        return NotImplementedError

    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass
        """
        return NotImplementedError


class BaseMultimodalModel(BaseModel):
    def predict(self, audios: List[np.ndarray], texts: List[str] = None) -> List[float]:
        return self.forward(audios, texts)

    def preprocess_audio(self, audios: np.ndarray) -> torch.Tensor:
        """
        Batched preprocessing (resampling, spectrogram etc.)
        """
        return NotImplementedError

    def preprocess_text(self, texts: List[str]) -> torch.Tensor:
        """
        Batched preprocessing (tokenization etc.)
        """
        return NotImplementedError

    def forward(self, audios: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass
        """
        return NotImplementedError
