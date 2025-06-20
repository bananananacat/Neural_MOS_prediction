import sys
import unittest
from pathlib import Path

# Добавляем корень проекта в sys.path **до** импортов сторонних пакетов
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../inference_test
sys.path.insert(0, str(PROJECT_ROOT))

import librosa  # noqa: E402
import torch  # noqa: E402
from src.models.mosnet import MosNet, MultiModalMosNet  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MOSNET_WEIGHTS = DATA_DIR / "MOSNet.pt"
MMOSNET_WEIGHTS = DATA_DIR / "MultiMOSNet.pt"
AUDIO_PATH = DATA_DIR / "LJ004-0088_032.wav"
TEXT_PATH = DATA_DIR / "example.txt"

EXPECTED_SCORE: float = 4.222222222222222


class TestMosNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio_model = MosNet(weights=MOSNET_WEIGHTS, device="cpu")
        cls.test_audio_path = AUDIO_PATH
        cls.expected_score = EXPECTED_SCORE

    def test_audio_model_initialization(self) -> None:
        self.assertIsNotNone(self.audio_model)
        self.assertEqual(self.audio_model.device, "cpu")

    def test_audio_preprocessing(self) -> None:
        audio, _ = librosa.load(self.test_audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).float()
        padded, masks = self.audio_model.preprocess_audios([audio_tensor])
        self.assertIsInstance(padded, torch.Tensor)
        self.assertIsInstance(masks, torch.Tensor)
        self.assertEqual(padded.dim(), 4)  # (B, C, T, F)

    def test_audio_prediction(self) -> None:
        audio, _ = librosa.load(self.test_audio_path, sr=16000)
        scores = self.audio_model.predict([audio])
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)
        self.assertGreaterEqual(scores[0], 1.0)
        self.assertLessEqual(scores[0], 5.0)


class TestMultiModalMosNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.multimodal_model = MultiModalMosNet(
            weights=MMOSNET_WEIGHTS,
            device="cpu",
        )
        cls.test_audio_path = AUDIO_PATH
        cls.test_text_path = TEXT_PATH
        cls.expected_score = EXPECTED_SCORE

    def test_multimodal_model_initialization(self) -> None:
        self.assertIsNotNone(self.multimodal_model)
        self.assertEqual(self.multimodal_model.device, "cpu")

    def test_multimodal_audio_preprocessing(self) -> None:
        audio, _ = librosa.load(self.test_audio_path, sr=16000)
        audio_tensor = self.multimodal_model.preprocess_audio([audio])
        self.assertIsInstance(audio_tensor, torch.Tensor)
        self.assertEqual(audio_tensor.dim(), 4)

    def test_text_preprocessing(self) -> None:
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        inputs = self.multimodal_model.preprocess_text([text])
        self.assertIsInstance(inputs, dict)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIsInstance(inputs["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs["attention_mask"], torch.Tensor)

    def test_multimodal_prediction(self) -> None:
        audio, _ = librosa.load(self.test_audio_path, sr=16000)
        with open(self.test_text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        scores = self.multimodal_model.predict([audio], [text])
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)
        self.assertGreaterEqual(scores[0], 1.0)
        self.assertLessEqual(scores[0], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
