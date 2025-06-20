# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.optim import AdamW
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange


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


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class CNN_BLSTM_MBNET2(nn.Module):
    def __init__(self, dropout=0.3):
        super(CNN_BLSTM_MBNET2, self).__init__()
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 3), 1), nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 3), 1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 3), 1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 3), 1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout)
        )
        # re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.droupout = nn.Dropout(dropout)
        # FC
        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)
        self.dense1 = nn.Sequential(
            TimeDistributed(nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.ReLU()), batch_first=True),
            nn.Dropout(dropout))

        # frame score
        self.frame_layer = TimeDistributed(nn.Linear(128, 1), batch_first=True)
        # avg score
        self.average_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, forward_input, mask):
        conv1_output = self.conv1(forward_input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)

        # reshape
        conv4_output = conv4_output.permute(0, 2, 1, 3)
        conv4_output = torch.reshape(conv4_output, (conv4_output.shape[0], conv4_output.shape[1], 4 * 128))

        # blstm
        blstm_output, (h_n, c_n) = self.blstm1(conv4_output)
        blstm_output = self.droupout(blstm_output)

        flatten_output = self.flatten(blstm_output)
        fc_output = self.dense1(flatten_output)
        frame_score = self.frame_layer(fc_output)

        # Применяем маску к frame_score
        frame_score = frame_score.squeeze(-1) * mask  # (batch, time)

        # Среднее значение только для валидных фреймов
        valid_sum = torch.sum(frame_score, dim=1)
        valid_count = torch.sum(mask, dim=1)
        avg_score = valid_sum / (valid_count + 1e-8)  # Избежание деления на 0

        return avg_score.unsqueeze(-1), frame_score


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    device = 'cuda'
    with torch.no_grad():
        for spectrograms, mos_scores, mask in tqdm(dataloader, desc="Validation"):
            spectrograms = spectrograms.to(device)
            mos_scores = mos_scores.to(device)
            mask = mask.to(device)
            avg_scores, _ = model(spectrograms, mask)
            loss = criterion(avg_scores, mos_scores)
            total_loss += loss.item() * spectrograms.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


if __name__ == "__main__":

    f = open("logW.txt", "a")
    f.write("Script started! ")
    f.close()

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

    device = torch.device("cuda")
    model = CNN_BLSTM_MBNET2().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 40
    best_val_loss = float("inf")

    f = open("logW.txt", "a")
    f.write("\n training started! ")
    f.close()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_list = []

        for spectrograms, mos_scores, mask in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            spectrograms = spectrograms.to(device)
            mos_scores = mos_scores.to(device)
            mask = mask.to(device)
            avg_scores, _ = model(spectrograms, mask)
            loss = criterion(avg_scores, mos_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
        avg_train_loss = torch.tensor(train_loss_list).mean().item()
        val_loss = evaluate_model(model, test_dataloader, criterion)

        val_loss = val_loss ** 0.5

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss and epoch >= 10:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join("model_checkpoints", f"AdamW_mod_{epoch}.pt"))

        f = open("logW.txt", "a")
        f.write(f" Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} ")
        f.close()

