import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

from utils.advanced_preprocessing import AdvancedHandwritingPreprocessor
from utils.advanced_feature_extraction import AdvancedFeatureExtractor
from utils.data_loader import EnhancedBalancedDataLoader


class EnhancedHandwritingTrainer:
    def __init__(self, data_path, use_ensemble=True):
        self.data_path = data_path
        self.use_ensemble = use_ensemble
        self.preprocessor = AdvancedHandwritingPreprocessor()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.data_loader = EnhancedBalancedDataLoader(target_samples_per_class=15)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')  # Для обработки NaN
        self.models = {}

    def load_and_balance_data(self):
        """Загрузка и балансировка данных с улучшенной обработкой"""
        print("=" * 60)
        print("УЛУЧШЕННАЯ ЗАГРУЗКА И БАЛАНСИРОВКА ДАННЫХ")
        print("=" * 60)

        # Загрузка с балансировкой
        self.features, self.labels = self.data_loader.load_balanced_dataset(
            self.data_path, self.preprocessor, self.feature_extractor
        )

        # Обработка NaN значений
        self._clean_features()

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

    def _clean_features(self):
        """Очистка признаков от NaN и бесконечных значений"""
        print("🧹 Очистка признаков от NaN и бесконечных значений...")

        # Заменяем бесконечности на NaN
        self.features = np.where(np.isfinite(self.features), self.features, np.nan)

        # Считаем количество NaN до очистки
        nan_count_before = np.isnan(self.features).sum()

        if nan_count_before > 0:
            print(f"  Найдено NaN значений: {nan_count_before}")

            # Используем импутер для заполнения NaN
            self.features = self.imputer.fit_transform(self.features)

            nan_count_after = np.isnan(self.features).sum()
            print(f"  NaN значений после очистки: {nan_count_after}")
        else:
            print("  NaN значений не обнаружено")

    def prepare_data_split(self):
        """Подготовка разделения данных с обработкой маленьких классов"""
        # Проверяем, есть ли классы с 1 образцом
        unique, counts = np.unique(self.labels_encoded, return_counts=True)
        min_samples = np.min(counts)

        if min_samples < 2:
            print(f"ВНИМАНИЕ: Минимальное количество образцов в классе: {min_samples}")
            print("Используем простое разделение без стратификации")

            # Простое разделение без стратификации
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.labels_encoded,
                test_size=0.2,
                random_state=42
            )
        else:
            # Стратифицированное разделение
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.labels_encoded,
                test_size=0.2,
                random_state=42,
                stratify=self.labels_encoded
            )

        return X_train, X_test, y_train, y_test

    def train_ensemble_model(self):
        """Обучение ансамблевой модели"""
        if not hasattr(self, 'features'):
            self.load_and_balance_data()

        # Разделение данных
        X_train, X_test, y_train, y_test = self.prepare_data_split()

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ АНСАМБЛЕВОЙ МОДЕЛИ")
        print("=" * 60)

        # Обучение нескольких моделей
        estimators = []

        # Random Forest
        print("🔨 Обучение Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        estimators.append(('rf', rf_model))

        # XGBoost
        print("🚀 Обучение XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        estimators.append(('xgb', xgb_model))

        # SVM с калибровкой вероятностей (только если нет NaN)
        if self.num_classes >= 2:  # SVM требует минимум 2 класса
            print("🎯 Обучение SVM...")
            try:
                svm_model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
                calibrated_svm = CalibratedClassifierCV(svm_model, method='sigmoid', cv=min(3, self.num_classes))
                calibrated_svm.fit(X_train_scaled, y_train)
                self.models['svm'] = calibrated_svm
                estimators.append(('svm', calibrated_svm))
                print("  SVM успешно обучен")
            except Exception as e:
                print(f"  ⚠️  SVM пропущен из-за ошибки: {e}")

        # Создание ансамбля
        if self.use_ensemble and len(estimators) > 1:
            print("🤝 Создание ансамбля...")
            try:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                ensemble.fit(X_train_scaled, y_train)
                self.models['ensemble'] = ensemble
                print("  Ансамбль успешно создан")
            except Exception as e:
                print(f"  ⚠️  Ансамбль не создан: {e}")

        # Кросс-валидация
        self.cross_validate_models(X_train_scaled, y_train)

        # Оценка на тестовых данных
        self.evaluate_all_models(X_test_scaled, y_test)

    def cross_validate_models(self, X, y, cv_folds=5):
        """Кросс-валидация всех моделей"""
        print(f"\n📊 КРОСС-ВАЛИДАЦИЯ ({min(cv_folds, self.num_classes)} фолдов)")
        print("-" * 40)

        # Убедимся, что количество фолдов не больше количества образцов в самом маленьком классе
        unique, counts = np.unique(y, return_counts=True)
        actual_cv = min(cv_folds, np.min(counts))

        if actual_cv < 2:
            print("⚠️  Слишком мало образцов для кросс-валидации")
            return

        skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            if name == 'ensemble' and len(self.models) == 1:
                continue

            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                print(f"{name:15}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"{name:15}: Ошибка валидации - {e}")

    def evaluate_all_models(self, X_test, y_test):
        """Оценка всех моделей"""
        print(f"\n📈 ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
        print("=" * 50)

        y_test_strings = self.label_encoder.inverse_transform(y_test)

        best_accuracy = 0
        best_model_name = ""

        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"\n{name}:")
                print(f"  Точность: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name

                # Детальная оценка для лучшей модели или при малом количестве классов
                if name == best_model_name or self.num_classes <= 10:
                    y_pred_strings = self.label_encoder.inverse_transform(y_pred)
                    print(f"\n  Детальный отчет:")
                    print(classification_report(y_test_strings, y_pred_strings, digits=4, zero_division=0))
            except Exception as e:
                print(f"\n{name}: Ошибка оценки - {e}")

        if best_model_name:
            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} (точность: {best_accuracy:.4f})")
            self.best_model_name = best_model_name
        else:
            print(f"\n❌ Ни одна модель не была успешно оценена")
            self.best_model_name = "random_forest"  # fallback

    def save_enhanced_models(self, save_path='models_advanced'):
        """Сохранение улучшенных моделей с метаданными"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Сохранение всех компонентов
        components = {
            'scaler.pkl': self.scaler,
            'label_encoder.pkl': self.label_encoder,
            'feature_extractor.pkl': self.feature_extractor,
            'feature_names.pkl': self.feature_names,
            'imputer.pkl': self.imputer  # Сохраняем импутер для предсказаний
        }

        for filename, obj in components.items():
            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(obj, f)

        # Сохранение всех моделей
        for name, model in self.models.items():
            with open(os.path.join(save_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)

        # Расширенная информация о датасете
        dataset_info = {
            'num_samples': len(self.features),
            'num_features': len(self.feature_names),
            'num_classes': self.num_classes,
            'class_names': list(self.label_encoder.classes_),
            'feature_names': self.feature_names,
            'best_model': getattr(self, 'best_model_name', 'random_forest'),
            'model_types': list(self.models.keys()),
            'training_timestamp': np.datetime64('now')
        }

        with open(os.path.join(save_path, 'dataset_info.pkl'), 'wb') as f:
            pickle.dump(dataset_info, f)

        print(f"\n💾 Улучшенные модели сохранены в {save_path}")
        print(f"   Сохранено моделей: {len(self.models)}")
        print(f"   Лучшая модель: {getattr(self, 'best_model_name', 'N/A')}")


def main():
    """Основная функция"""
    try:
        print("ИСПРАВЛЕННАЯ УЛУЧШЕННАЯ СИСТЕМА АНАЛИЗА ПОЧЕРКА")
        print("=" * 60)

        data_path = 'data/rulers'

        trainer = EnhancedHandwritingTrainer(data_path)
        trainer.load_and_balance_data()
        trainer.train_ensemble_model()
        trainer.save_enhanced_models()

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()