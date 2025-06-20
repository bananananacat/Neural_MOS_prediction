from typing import Any, List, Tuple

from einops import rearrange
import librosa
import numpy as np

from src.models.base_model import BaseModel, BaseMultimodalModel

import torch
import torch.nn.functional as f
from torch import nn

from transformers import BertModel, BertTokenizer


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool) -> None:
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class CnnBlstmMbnet2(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout),
        )
        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.droupout = nn.Dropout(dropout)
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributed(
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                ),
                batch_first=True,
            ),
            nn.Dropout(dropout),
        )
        self.frame_layer = TimeDistributed(nn.Linear(128, 1), batch_first=True)
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, forward_input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv4_output = conv4_output.permute(0, 2, 1, 3)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], conv4_output.shape[1], 4 * 128))
        blstm_output, _ = self.blstm1(conv4_output)
        blstm_output = self.droupout(blstm_output)
        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)
        frame_score = self.frame_layer(fc_output)
        frame_score = frame_score.squeeze(-1) * mask
        valid_sum = torch.sum(frame_score, dim=1)
        valid_count = torch.sum(mask, dim=1)
        avg_score = valid_sum / (valid_count + 1e-8)
        return avg_score.unsqueeze(-1), frame_score


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_, gate = x.chunk(2, dim=-1)
        return f.silu(gate) * x_


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, scale_base: int = 512, use_xpos: bool = True) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor, scale: float = 1.) -> torch.Tensor:
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return f.normalize(t, dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim_head: int = 64, heads: int = 8, dropout: float = 0.2, forward_expansion: int = 2, device: str = "cpu") -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.embed_dim = heads * dim_head
        self.device = device

        self.qkv = nn.Linear(dim_head * heads, dim_head * heads * 3)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.norm = nn.LayerNorm(dim_head * heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_head * heads, forward_expansion * dim_head * heads * 2),  # *2 для SwiGLU
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * dim_head * heads, dim_head * heads),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        n, seq_length, _ = q.shape
        qkv_proj = self.qkv(q)
        qkv_proj = qkv_proj.reshape(n, seq_length, self.heads, 3 * self.dim_head)
        qkv = qkv_proj.permute(0, 2, 1, 3)
        q_, k_, v_ = qkv.chunk(3, dim=-1)
        q_, k_ = map(l2norm, (q_, k_))
        q_ = q_ * self.q_scale
        k_ = k_ * self.k_scale
        positions, scale = self.rotary_emb(seq_length, self.device)
        q_ = apply_rotary_pos_emb(positions, q_, scale)
        k_ = apply_rotary_pos_emb(positions, k_, scale ** -1)
        attn_output = f.scaled_dot_product_attention(q_, k_, v_)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(n, seq_length, self.embed_dim)
        attn_output = self.norm(attn_output)
        forward_output = self.feed_forward(attn_output)
        return attn_output + forward_output


class AudioFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), padding=1), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), padding=1), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), padding=1), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), padding=1), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class CrossAttentionModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.audio_extractor = AudioFeatureExtractor()

        self.text_projection = nn.Linear(768, 512)
        # передаём device внутрь TransformerBlock
        self.cross_attention = TransformerBlock(dim_head=64, heads=8, device=device)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.frame_layer = nn.Linear(128, 1)
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        audio_input: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """audio_input  shape: (B, 1, T, F)
           text_embeddings shape: (B, 768)
        """
        # ↳ Audio branch
        audio_features = self.audio_extractor(audio_input)          # (B, T, 512)

        # ↳ Text branch
        text_proj = self.text_projection(text_embeddings)           # (B, 512)
        text_proj = text_proj.unsqueeze(1)                          # (B, 1, 512)

        # Cross-attention
        cross_out = self.cross_attention(audio_features, text_proj, text_proj)  # (B, T, 512)

        # Head
        fc_out = self.fc1(cross_out)                                # (B, T, 128)
        frame_score = self.frame_layer(fc_out)                      # (B, T, 1)

        # aggregate
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))  # (B, 1, 1)
        return avg_score.reshape(avg_score.size(0), -1), frame_score.squeeze()


class MosNet(BaseModel):
    def __init__(self, weights: str, device: str = "cpu") -> None:
        self.device = device
        self.model = self._load_weights(weights)
        self.sample_rate = 16000
        self.fft_size = 512
        self.hop_length = 256
        self.win_length = 512

    def _load_weights(self, weights: str) -> torch.nn.Module:
        model = CnnBlstmMbnet2()
        state_dict = torch.load(weights, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_audios(self, audios: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrograms = []
        for audio in audios:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float().to(self.device)
            else:
                audio_tensor = audio.float().to(self.device)
            audio_np = audio_tensor.cpu().numpy()
            spec = librosa.stft(audio_np, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length)
            mag = np.abs(spec).astype(np.float32).T
            mag_tensor = torch.tensor(mag, device=self.device).unsqueeze(0)
            spectrograms.append(mag_tensor)
        max_len = max(spec.shape[1] for spec in spectrograms)
        batch_size, feat_dim = len(spectrograms), spectrograms[0].shape[2]
        padded = torch.zeros(batch_size, 1, max_len, feat_dim, device=self.device)
        masks = torch.zeros(batch_size, max_len, device=self.device)
        for i, spec in enumerate(spectrograms):
            valid_len = spec.shape[1]
            padded[i, :, :valid_len, :] = spec
            masks[i, :valid_len] = 1.0
        return padded, masks

    def forward(self, audios: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.model(audios.to(self.device), masks.to(self.device))
        return outputs

    def predict(self, audios: List[Any]) -> List[float]:
        with torch.no_grad():
            padded, masks = self.preprocess_audios(audios)
            scores = self.forward(padded, masks)
        return scores.squeeze(-1).cpu().tolist()


class MultiModalMosNet(BaseMultimodalModel):
    def __init__(self, weights: str, device: str = "cpu") -> None:
        self.device = device
        self.model = self._load_weights(weights)
        self.sample_rate = 16000
        self.fft_size = 512
        self.hop_length = 256
        self.win_length = 512

    def _load_weights(self, weights: str) -> torch.nn.Module:
        model = CrossAttentionModel(device=self.device)
        state_dict = torch.load(weights, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_audio(self, audios: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for audio in audios:
            y = torch.tensor(audio, dtype=torch.float32, device=self.device)
            spec = torch.stft(y, n_fft=512, hop_length=256, win_length=512, return_complex=False)
            mag = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)
            mag = mag.permute(1, 0).unsqueeze(0)
            tensors.append(mag)
        max_len = max(t.shape[1] for t in tensors)
        padded = torch.zeros(len(tensors), 1, max_len, tensors[0].shape[2], device=self.device)
        for i, t in enumerate(tensors):
            padded[i, :, :t.shape[1], :] = t
        return padded

    def preprocess_text(self, texts: List[str]) -> dict:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        with torch.no_grad():
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def forward(self, audios: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        avg_score, _ = self.model(audios.to(self.device), texts.to(self.device))
        return avg_score

    def predict(self, audios: List[np.ndarray], texts: List[str] = None) -> List[float]:
        with torch.no_grad():
            audios_tensor = self.preprocess_audio(audios)
            inputs = self.preprocess_text(texts)
            model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
            model.eval()
            outputs = model(**inputs)
            texts_tensor = outputs.last_hidden_state[:, 0, :]
            preds = self.forward(audios_tensor, texts_tensor)
            result = preds.squeeze().cpu().tolist()
            if isinstance(result, float):
                return [result]
            return [float(x) for x in result]
