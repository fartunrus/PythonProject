import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils.preprocessing import HandwritingPreprocessor
from utils.feature_extraction import HandwritingFeatureExtractor
from utils.data_loader import BalancedDataLoader

class SimpleHandwritingTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = HandwritingPreprocessor()
        self.feature_extractor = HandwritingFeatureExtractor()
        self.data_loader = BalancedDataLoader(target_samples_per_class=10)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}

    def load_and_balance_data(self):
        """Загрузка и балансировка данных"""
        print("=" * 60)
        print("ЗАГРУЗКА И БАЛАНСИРОВКА ДАННЫХ")
        print("=" * 60)

        # Загрузка с балансировкой
        self.features, self.labels = self.data_loader.load_balanced_dataset(
            self.data_path, self.preprocessor, self.feature_extractor
        )

        # Кодирование меток
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.feature_names = self.feature_extractor.feature_names

        print(f"\nФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"Общее количество образцов: {len(self.features)}")
        print(f"Количество классов: {self.num_classes}")
        print(f"Количество признаков: {len(self.feature_names)}")

        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} образцов")

    def train_simple_model(self):
        """Простое обучение модели"""
        if not hasattr(self, 'features'):
            self.load_and_balance_data()

        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=self.labels_encoded
        )

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)

        # Random Forest
        print("Обучение Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model

        # Оценка модели
        self.evaluate_model(X_test_scaled, y_test)

    def evaluate_model(self, X_test, y_test):
        """Оценка модели"""
        model = self.models['random_forest']
        y_pred = model.predict(X_test)
        y_test_strings = self.label_encoder.inverse_transform(y_test)
        y_pred_strings = self.label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nТочность модели: {accuracy:.4f}")

        if self.num_classes <= 10:
            print(f"\nОтчет по классам:")
            print(classification_report(y_test_strings, y_pred_strings, digits=4))

    def save_models(self, save_path='models'):
        """Сохранение моделей"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Сохранение компонентов
        components = {
            'scaler.pkl': self.scaler,
            'label_encoder.pkl': self.label_encoder,
            'feature_extractor.pkl': self.feature_extractor,
        }

        for filename, obj in components.items():
            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(obj, f)

        # Сохранение модели
        with open(os.path.join(save_path, 'random_forest.pkl'), 'wb') as f:
            pickle.dump(self.models['random_forest'], f)

        # Сохранение информации о датасете (ДОБАВЛЕНО)
        dataset_info = {
            'num_samples': len(self.features),
            'num_features': len(self.feature_names),
            'num_classes': self.num_classes,
            'class_names': list(self.label_encoder.classes_),
            'feature_names': self.feature_names
        }

        with open(os.path.join(save_path, 'dataset_info.pkl'), 'wb') as f:
            pickle.dump(dataset_info, f)

        print(f"\nМодели сохранены в {save_path}")
        print(f"Информация о датасете: {dataset_info}")

def main():
    """Основная функция"""
    try:
        print("УПРОЩЕННАЯ СИСТЕМА АНАЛИЗА ПОЧЕРКА")
        print("=" * 60)

        data_path = 'data/rulers'

        trainer = SimpleHandwritingTrainer(data_path)
        trainer.train_simple_model()
        trainer.save_models()

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("=" * 60)

    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()