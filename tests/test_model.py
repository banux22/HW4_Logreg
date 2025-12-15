import pytest
import pandas as pd
import numpy as np
from ml_logreg_hw.model import DiabetesClassifier

@pytest.fixture
def raw_data():
    """Создает мини-датасет, похожий на Pima Indians Diabetes."""
    data = {
        'Pregnancies': [1, 0, 8, 1, 0],
        'Glucose': [85, 0, 183, 89, 137],  # Есть 0, который нужно заменить
        'BloodPressure': [66, 0, 64, 66, 40], # Есть 0
        'SkinThickness': [29, 0, 0, 23, 35],  # Есть 0
        'Insulin': [0, 0, 0, 94, 168],        # Есть 0
        'BMI': [26.6, 0, 23.3, 28.1, 43.1],   # Есть 0
        'DiabetesPedigreeFunction': [0.35, 0.2, 0.67, 0.16, 2.28],
        'Age': [31, 20, 32, 21, 33],
        'Outcome': [0, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_preprocessing_zeros(raw_data):
    """Проверяет, что 0 заменяются и пропуски заполняются."""
    clf = DiabetesClassifier()
    X_processed, y = clf.preprocess_data(raw_data, is_training=True)
    
    # 1. Проверка размерности
    assert X_processed.shape == (5, 8), "Размерность матрицы X неверна"
    
    # 2. Проверка, что нет пропусков (NaN)
    assert not np.isnan(X_processed).any(), "В обработанных данных остались NaN"
    
    # 3. Проверка масштабирования (среднее должно быть ~0, std ~1)
    # Проверяем на колонке Glucose (индекс 1)
    col_mean = X_processed[:, 1].mean()
    col_std = X_processed[:, 1].std()
    
    assert np.isclose(col_mean, 0, atol=1e-1), "Данные не масштабированы (Mean != 0)"
    assert np.isclose(col_std, 1, atol=1e-1), "Данные не масштабированы (Std != 1)"

def test_training_flow(raw_data):
    """Проверяет, что модель обучается и делает предсказания."""
    clf = DiabetesClassifier()
    clf.train(raw_data)
    
    # Проверяем, что модель sklearn инициализирована и обучена
    assert hasattr(clf.model, "coef_"), "Модель не обучена (нет весов)"
    
    # Проверяем предсказание
    preds = clf.predict(raw_data)
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1}), "Предсказания должны быть 0 или 1"

def test_metrics_calculation(raw_data):
    """Проверяет расчет метрик."""
    clf = DiabetesClassifier()
    clf.train(raw_data)
    
    metrics = clf.evaluate(raw_data)
    
    required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for m in required_metrics:
        assert m in metrics, f"Метрика {m} отсутствует в отчете"
        assert 0 <= metrics[m] <= 1, f"Метрика {m} выходит за пределы [0, 1]"

def test_data_leakage_prevention(raw_data):
    """
    Проверяет, что scaler не переобучается на тестовых данных.
    Среднее значение scaler должно остаться таким же, как на train.
    """
    clf = DiabetesClassifier()
    
    # Тренируем на части данных
    train_df = raw_data.iloc[:3]
    test_df = raw_data.iloc[3:]
    
    clf.train(train_df)
    saved_mean = clf.scaler.mean_[1] # Среднее для Glucose после train
    
    # Делаем предсказание на тесте
    clf.predict(test_df)
    
    # Проверяем, что среднее в скейлере НЕ изменилось
    current_mean = clf.scaler.mean_[1]
    
    assert saved_mean == current_mean, "Data Leakage! Скейлер переобучился на тестовых данных"
