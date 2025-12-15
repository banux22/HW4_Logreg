import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

class DiabetesClassifier:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        """
        Инициализация классификатора.
        :param C: Обратная сила регуляризации (чем меньше, тем сильнее штраф).
        """
        self.C = C
        self.random_state = random_state
        
        self.model = LogisticRegression(C=self.C, random_state=self.random_state, max_iter=1000)
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        ЭТАП 1: Очистка и Масштабирование.
        
        Задачи:
        1. Найти колонки, где 0 быть не может (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
           Заменить в них 0 на NaN.
        2. Заполнить пропуски медианой (используя self.imputer).
           ВАЖНО: Если is_training=True, делаем fit_transform, иначе transform.
        3. Масштабировать признаки (используя self.scaler).
           ВАЖНО: Если is_training=True, делаем fit_transform, иначе transform.
        4. Вернуть X (признаки) и y (целевую переменную 'Outcome').
        """
        data = df.copy()
        
        # 1. Замена нулей на NaN в физиологических признаках
        # TODO: Реализовать замену 0 -> np.nan
        
        # Разделение на X и y
        target_col = 'Outcome'
        if target_col in data.columns:
            y = data[target_col].values
            X_raw = data.drop(columns=[target_col])
        else:
            y = None
            X_raw = data
            
        # 2. Заполнение пропусков (Imputation)
        # TODO: Реализовать логику fit_transform (если is_training) / transform
        
        # 3. Масштабирование (Scaling)
        # TODO: Реализовать логику fit_transform (если is_training) / transform
        
        # Заглушка, чтобы код не падал до реализации
        X_processed = np.array([]) 
        
        return X_processed, y

    def train(self, df_train: pd.DataFrame):
        """
        ЭТАП 2: Обучение модели.
        1. Предобработать данные (is_training=True).
        2. Обучить self.model.
        """
        # TODO: Ваш код здесь
        pass

    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        ЭТАП 3: Предсказание классов (0 или 1).
        1. Предобработать данные (is_training=False).
        2. Вернуть предсказания.
        """
        # TODO: Ваш код здесь
        return np.array([])

    def predict_proba(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        ЭТАП 4: Предсказание вероятностей (для ROC-AUC).
        Возвращает вероятность класса 1.
        """
        # TODO: Ваш код здесь
        return np.array([])

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """
        ЭТАП 5: Оценка качества.
        Возвращает словарь с метриками:
        - Accuracy
        - Precision
        - Recall
        - F1
        - ROC-AUC
        """
        # TODO: Получить предсказания и сравнить с реальным y из df_test
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0
        }
