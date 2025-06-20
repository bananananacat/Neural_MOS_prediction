'''
Полный скрипт обучения модели AttentionMOS - он отличается от MultiModal MOSNet тем, что вместо текстовых матриц в матрицы K и V передается аудио.
'''

# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pandas.core import frame
import os
import time
import librosa
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import math
import os
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange


class MOSDataset(Dataset):
    def __init__(self, audio_dir, txt_path, sample_rate=16000, fft_size=512, hop_length=256, win_length=512):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.data = self._parse_txt_file(txt_path)

    def _parse_txt_file(self, txt_path):
        data = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as file:
                lines = file.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(',')
                    utterance_id = parts[1].strip()
                    mean_score = float(parts[2].strip())

                    wav_path = os.path.join(self.audio_dir, utterance_id)
                    if os.path.exists(wav_path):
                        y, _ = librosa.load(wav_path, sr=self.sample_rate)
                        duration = len(y) / self.sample_rate
                        data.append((utterance_id, mean_score, duration))

        # Сортировка по длительности
        data.sort(key=lambda x: x[2])
        return [(d[0], d[1]) for d in data]

    def __len__(self):
        return len(self.data)

    def get_spectrogram(self, wav_path):
        if not os.path.exists(wav_path):
            return None
        y, _ = librosa.load(wav_path, sr=self.sample_rate)

        linear = librosa.stft(y, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length)
        mag = np.abs(linear).astype(np.float32)

        # (T, F) -> (1, T, F)
        mag = np.expand_dims(mag.T, axis=0)
        return torch.tensor(mag)

    def __getitem__(self, idx):
        utterance_id, mos_score = self.data[idx]
        wav_path = os.path.join(self.audio_dir, utterance_id)
        spectrogram = self.get_spectrogram(wav_path)

        mask = torch.ones(spectrogram.shape[-2], dtype=torch.float32)
        mos_score = torch.tensor([mos_score], dtype=torch.float32)
        return spectrogram, mos_score, mask


def collate_fn(batch):
    #:return: padded spectrogram tensor, MOS tensor, mask tensor
    spectrograms, mos_scores, masks = zip(*batch)  # Теперь учитываем третье значение

    # max time len
    max_len = max(spectrogram.shape[1] for spectrogram in spectrograms)

    # Создаем тензор для паддинга спектрограммы
    padded_spectrograms = torch.zeros(len(spectrograms), 1, max_len, spectrograms[0].shape[2])

    # Создаем маску: 1 для валидных фреймов, 0 для паддингов
    padded_masks = torch.zeros(len(spectrograms), max_len)

    for i, spectrogram in enumerate(spectrograms):
        valid_len = spectrogram.shape[1]
        padded_spectrograms[i, :, :valid_len, :] = spectrogram
        padded_masks[i, :valid_len] = masks[i]  # Копируем маску для валидных фреймов

    # MOS оценки
    mos_scores = torch.stack(mos_scores)

    return padded_spectrograms, mos_scores, padded_masks


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base=512, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale=1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


def l2norm(t):
    return F.normalize(t, dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim_head=64, heads=8, dropout=0.2, forward_expansion=2, device="cuda"):
        super(TransformerBlock, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.embed_dim = heads * dim_head
        self.device = device

        self.qkv = nn.Linear(dim_head * heads, dim_head * heads * 3)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.rotary_emb = RotaryEmbedding(dim_head)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim_head * heads * forward_expansion

        self.norm = nn.LayerNorm(dim_head * heads)
        self.feed_forward = nn.Sequential(
            # Изменено: убрано умножение на 2 и SwiGLU заменен на SiLU
            nn.Linear(dim_head * heads, forward_expansion * dim_head * heads),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * dim_head * heads, dim_head * heads),
        )

    def forward(self, Q, K, V):
        N, seq_length, _ = Q.shape

        qkv_proj = self.qkv(Q)
        qkv_proj = qkv_proj.reshape(N, seq_length, self.heads, 3 * self.dim_head)
        qkv = qkv_proj.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        positions, scale = self.rotary_emb(seq_length, self.device)
        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, seq_length, self.embed_dim)

        attn_output = self.norm(attn_output)
        forward_output = self.feed_forward(attn_output)
        return attn_output + forward_output


# Остальные части модели (AudioFeatureExtractor и CrossAttentionModel) остаются без изменений

class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        return x


class CrossAttentionModel(nn.Module):
    def __init__(self):
        super(CrossAttentionModel, self).__init__()
        self.audio_extractor = AudioFeatureExtractor()
        self.text_projection = nn.Linear(768, 512)

        self.cross_attention = TransformerBlock(dim_head=64, heads=8)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.frame_layer = nn.Linear(128, 1)
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, audio_input, mask):
        audio_features = self.audio_extractor(audio_input)
        bs, T, CxF = audio_features.shape

        if not hasattr(self, 'audio_projection'):
            self.audio_projection = nn.Linear(CxF, 512).to(audio_features.device)

        audio_features = self.audio_projection(audio_features)
        #text_embeddings_projected = self.text_projection(text_embeddings)
        #text_embeddings_projected = text_embeddings_projected.unsqueeze(1)

        cross_attention_output = self.cross_attention(audio_features, audio_features,
                                                      audio_features)

        fc_output = self.fc1(cross_attention_output)
        frame_score = self.frame_layer(fc_output)
        avg_score = self.average_layer(frame_score.permute(0, 2, 1))

        return torch.reshape(avg_score, (avg_score.shape[0], -1)), frame_score.squeeze()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, avg_scores, mos_scores, frame_scores):
        mos_scores = mos_scores.squeeze()
        frame_scores = frame_scores.squeeze()
        avg_scores = avg_scores.squeeze()
        # Utterance-level MSE
        utterance_loss = self.mse_loss(avg_scores, mos_scores)

        mos_scores_expanded = mos_scores.unsqueeze(1).expand(-1, frame_scores.size(1))

        # Frame-level MSE (без маски)
        #print(frame_scores.shape, mos_scores_expanded.shape)
        frame_mse_loss = (frame_scores - mos_scores_expanded) ** 2
        frame_loss = frame_mse_loss.mean(dim=1).mean()
        total_loss = utterance_loss + self.alpha * frame_loss
        return total_loss


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for spectrograms, mos_scores, text_embedding in tqdm(dataloader, desc="Validation"):
            text_embedding = text_embedding.to(device)
            spectrograms = spectrograms.to(device)
            mos_scores = mos_scores.to(device)
            avg_scores, _ = model(spectrograms, text_embedding)
            loss = criterion(avg_scores, mos_scores)
            total_loss += loss.item() * spectrograms.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

if __name__ == "__main__":

    f = open("atn.txt", "a")
    f.write("Script started! ")
    f.close()

    train_dataset = MOSDataset(
        audio_dir="update_SOMOS_v2/all_audios/all_wavs",
        txt_path="update_SOMOS_v2/training_files_with_SBS/training_files/clean/train_mos_list.txt",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512
    )

    test_dataset = MOSDataset(
        audio_dir="update_SOMOS_v2/all_audios/all_wavs",
        txt_path="update_SOMOS_v2/training_files_with_SBS/training_files/clean/test_mos_list.txt",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(len(train_dataset), len(test_dataset))

    device = torch.device("cuda")
    model = CrossAttentionModel().to(device)
    criterion = CombinedLoss(alpha=1.0)
    metric = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    num_epochs = 50
    best_val_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    f = open("atn.txt", "a")
    f.write("\n training started! ")
    f.close()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_list = []
        for audio_inputs, mos_scores, mask in train_dataloader:
            audio_inputs = audio_inputs.to(device)
            mos_scores = mos_scores.to(device)
            mask = mask.to(device)
            avg_scores, frame_scores = model(audio_inputs, mask)
            # print(avg_scores, frame_scores)
            loss = criterion(avg_scores, mos_scores, frame_scores)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        avg_train_loss = torch.tensor(train_loss_list).mean().item()
        val_loss = evaluate_model(model, test_dataloader, metric)
        val_loss = val_loss ** 0.5
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join("model_checkpoints", f"AttentionModel_best{epoch}.pt"))

        scheduler.step(val_loss)

        f = open("atn.txt", "a")
        f.write(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}"     )
        f.close()
