# Neural_MOS_prediction

ФИО: Левин Марк Романович

Проект: Модель оценки TTS метрик.

В данном репозитории вы можете найти код моделей, предсказывающих MOS оценку аудио, а также различные эксперименты.

Jupyter Notebook со всеми экспериментами лежит вот [здесь](https://github.com/bananananacat/HSE_DL_2025/blob/main/project/MOSNET.ipynb)

Класс модели лежит [здесь](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/src/models/mosnet.py) (от отнаследован от [класса](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/src/models/base_model.py) в этой же директории)

Также есть [тесты](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/tests/test_models.py). Для запуска тестов и моделей надо скачать веса предобученных моделей с [гугл диска](https://drive.google.com/drive/folders/1iOhUhGE3fG4phKu73qhw7SV2s5fMIFAq) и поместить в папку [data](https://github.com/bananananacat/Neural_MOS_prediction/tree/main/tests/data), где уже лежит пример аудио и соответствующего текста.
