'''
Полный скрипт обучения CNNMOS - причем закомментированный класс датасет относится к старой версии SOMOS, а незакомментированный - к std SOMOS
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
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from einops import rearrange
from torch.optim import AdamW


'''
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
                        # Определение длительности аудио
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
    spectrograms, mos_scores, masks = zip(*batch)

    max_len = max(spectrogram.shape[1] for spectrogram in spectrograms)
    padded_spectrograms = torch.zeros(len(spectrograms), 1, max_len, spectrograms[0].shape[2])

    padded_masks = torch.zeros(len(spectrograms), max_len)

    for i, spectrogram in enumerate(spectrograms):
        valid_len = spectrogram.shape[1]
        padded_spectrograms[i, :, :valid_len, :] = spectrogram
        padded_masks[i, :valid_len] = masks[i]
    mos_scores = torch.stack(mos_scores)
    return padded_spectrograms, mos_scores, padded_masks
'''
class MOSDataset(Dataset):
    def __init__(self, audio_dir, csv_path, sample_rate=16000, fft_size=512, hop_length=256, win_length=512,
                 mode='train'):
        """
        Args:
            audio_dir (str): Путь к директории с аудиофайлами.
            csv_path (str): Путь к CSV-файлу с метаданными.
            sample_rate (int): Частота дискретизации аудио.
            fft_size (int): Размер FFT для спектрограммы.
            hop_length (int): Шаг для спектрограммы.
            win_length (int): Длина окна для спектрограммы.
            mode (str): Режим работы ('train' или 'test').
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.mode = mode
        self.data = self._parse_csv_file(csv_path)

    def _parse_csv_file(self, csv_path):
        data = []
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, delimiter=',', header=0, dtype=str)  # Читаем CSV с запятой как разделитель

            for _, row in df.iterrows():
                utterance_id = row.iloc[0].strip() + ".wav"  # Добавляем .wav к названию файла
                try:
                    mean_score = float(row.iloc[1])  # MOS-оценка из второго столбца
                except ValueError:
                    continue  # Пропускаем строки с некорректными значениями

                # Фильтрация по режиму (train или test)
                set_value = int(row.iloc[4])  # Столбец 'set' (0 для train, 1 для test)
                if (self.mode == 'train' and set_value == 0) or (self.mode == 'test' and set_value == 1):
                    wav_path = os.path.join(self.audio_dir, utterance_id)
                    if os.path.exists(wav_path):
                        # Определение длительности аудио
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
    spectrograms, mos_scores, masks = zip(*batch)

    max_len = max(spectrogram.shape[1] for spectrogram in spectrograms)
    padded_spectrograms = torch.zeros(len(spectrograms), 1, max_len, spectrograms[0].shape[2])

    padded_masks = torch.zeros(len(spectrograms), max_len)

    for i, spectrogram in enumerate(spectrograms):
        valid_len = spectrogram.shape[1]
        padded_spectrograms[i, :, :valid_len, :] = spectrogram
        padded_masks[i, :valid_len] = masks[i]
    mos_scores = torch.stack(mos_scores)
    return padded_spectrograms, mos_scores, padded_masks


class CNN_MOS(nn.Module):
    def __init__(self, dropout=0.3):
        super(CNN_MOS, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(16), nn.Dropout(dropout),

            nn.Conv2d(16, 32, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(32), nn.Dropout(dropout),

            nn.Conv2d(32, 64, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(64), nn.Dropout(dropout),

            nn.Conv2d(64, 128, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(128), nn.Dropout(dropout)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 257, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        #print(x.shape)
        x = self.conv_layers(x)
        #print(x.shape)
        x = x.permute(0, 2, 1, 3)
        #print(x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        #print(x.shape)
        frame_scores = self.fc_layers(x).squeeze(-1) * mask
        avg_score = frame_scores.sum(dim=1) / (mask.sum(dim=1))
        return avg_score.unsqueeze(-1), frame_scores


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    device = 'cuda'
    with torch.no_grad():
        for spectrograms, mos_scores, mask in dataloader:
            spectrograms, mos_scores, mask = spectrograms.to(device), mos_scores.to(device), mask.to(device)
            avg_scores, _ = model(spectrograms, mask)
            loss = criterion(avg_scores, mos_scores)
            total_loss += loss.item() * spectrograms.size(0)
    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":

    f = open("cnn_mos.txt", "a")
    f.write("Script started! ")
    f.close()
    '''
    train_dataset = MOSDataset(
        audio_dir="/home/mrlevin/update_SOMOS_v2/all_audios/all_wavs",
        txt_path="/home/mrlevin/update_SOMOS_v2/training_files_with_SBS/training_files/clean/train_mos_list.txt",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512
    )

    test_dataset = MOSDataset(
        audio_dir="/home/mrlevin/update_SOMOS_v2/all_audios/all_wavs",
        txt_path="/home/mrlevin/update_SOMOS_v2/training_files_with_SBS/training_files/clean/test_mos_list.txt",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512
    )
    '''
    train_dataset = MOSDataset(
        audio_dir="/home/mrlevin/update_SOMOS_v2/all_audios/all_wavs",
        csv_path="/home/mrlevin/david_test.csv",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512,
        mode='train'
    )

    test_dataset = MOSDataset(
        audio_dir="/home/mrlevin/update_SOMOS_v2/all_audios/all_wavs",
        csv_path="/home/mrlevin/david_test.csv",
        sample_rate=16000,
        fft_size=512,
        hop_length=256,
        win_length=512,
        mode='test'
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    f = open("cnn_mos.txt", "a")
    f.write("Dataloaders created! ")
    f.close()
    device = torch.device("cuda")
    model = CNN_MOS().to(device)

    f = open("cnn_mos.txt", "a")
    f.write("model initialized! ")
    f.close()

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 40
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # Косинусный планировщик

    f = open("cnn_mos.txt", "a")
    f.write("training started! ")
    f.close()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_list = []
        t = time.time()
        for spectrograms, mos_scores, mask in train_dataloader:
            spectrograms, mos_scores, mask = spectrograms.to(device), mos_scores.to(device), mask.to(device)
            avg_scores, _ = model(spectrograms, mask)
            loss = criterion(avg_scores, mos_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
        scheduler.step()
        mse = evaluate_model(model, test_dataloader, criterion)
        rmse = mse**0.5
        mean_loss = np.mean(train_loss_list)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {mean_loss:.4f} - Val Loss: {rmse:.4f}")
        f = open("cnn_mos.txt", "a")
        f.write(f"Epoch {epoch}/{num_epochs} - Train Loss: {mean_loss:.4f} - Val Loss: {rmse:.4f}  ")
        f.close()
        print("time taken: ", t-time.time())
    torch.save(model.state_dict(), os.path.join("/home/mrlevin/model_checkpoints", 'CNN_MOS.pt'))


