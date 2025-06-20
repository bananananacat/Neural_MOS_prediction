# Neural_MOS_prediction

ФИО: Левин Марк Романович

Проект: Модель оценки TTS метрик.

В данном репозитории вы можете найти код моделей, предсказывающих MOS оценку аудио, а также различные эксперименты с моими архитектурами. Через какое-то время выйдет статья на Хабре, а веса моделей появятся на HuggingFace(приложу все ссылки сюда).

Jupyter Notebook с некоторыми экспериментами лежит вот [здесь](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/MOSNET.ipynb) (здесь не все, большая часть экспериментов проводилась на суперкомпьютере).

Классы моделей MOSNet и MultiModal MOSNet лежат [здесь](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/src/models/mosnet.py) (отнаследованы от [класса](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/src/models/base_model.py) в этой же директории).

Полные скрипты обучения моделей MOSNet и MultiModal MOSNet лежат [здесь](https://github.com/bananananacat/Neural_MOS_prediction/tree/main/src/models/other_scripts).

Полные скрипты обучения менее хороших моделей AttentionMOS и CNNMOS лежат [здесь](https://github.com/bananananacat/Neural_MOS_prediction/tree/main/src/other_models).

Также есть [тесты](https://github.com/bananananacat/Neural_MOS_prediction/blob/main/tests/test_models.py). Для запуска тестов и моделей надо скачать веса предобученных моделей с [гугл диска](https://drive.google.com/drive/folders/1iOhUhGE3fG4phKu73qhw7SV2s5fMIFAq) и поместить в папку [data](https://github.com/bananananacat/Neural_MOS_prediction/tree/main/tests/data), где уже лежит пример аудио и соответствующего текста.

Стандартизованный датасет SOMOS, менее учитывающий субъективность оценок, лежит там же, где находятся веса моделей.
