import os
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils.preprocessing import HandwritingPreprocessor
from utils.feature_extraction import HandwritingFeatureExtractor


class HandwritingPredictor:
    def __init__(self, models_path='models'):
        self.models_path = models_path
        self.preprocessor = HandwritingPreprocessor()
        self.load_models()

    def load_models(self):
        """Загрузка обученных моделей и компонентов"""
        try:
            print("Загрузка моделей...")

            # Проверяем существование папки models
            if not os.path.exists(self.models_path):
                raise FileNotFoundError(f"Папка {self.models_path} не найдена! Сначала выполните обучение.")

            # Загрузка scaler
            scaler_path = os.path.join(self.models_path, 'scaler.pkl')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Файл {scaler_path} не найден!")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Загрузка label encoder
            label_encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(f"Файл {label_encoder_path} не найден!")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            # Загрузка feature extractor
            feature_extractor_path = os.path.join(self.models_path, 'feature_extractor.pkl')
            if not os.path.exists(feature_extractor_path):
                raise FileNotFoundError(f"Файл {feature_extractor_path} не найден!")
            with open(feature_extractor_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)

            # Загрузка модели Random Forest
            model_path = os.path.join(self.models_path, 'random_forest.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл {model_path} не найден!")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Загрузка информации о датасете (если есть)
            dataset_info_path = os.path.join(self.models_path, 'dataset_info.pkl')
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'rb') as f:
                    self.dataset_info = pickle.load(f)
                print("✅ Информация о датасете загружена")
            else:
                self.dataset_info = None
                print("⚠️  Информация о датасете не найдена (dataset_info.pkl)")

            print("✅ Модели и компоненты успешно загружены")
            print(f"✅ Классы в модели: {list(self.label_encoder.classes_)}")

        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            print("\n💡 Решение: Запустите training_simple.py для обучения моделей")
            raise

    def predict_handwriting(self, image_path, top_k=5):
        """
        Предсказание схожести почерка с правителями
        """
        try:
            print(f"\n🔍 Анализ изображения: {os.path.basename(image_path)}")

            # Проверяем существование файла
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Изображение {image_path} не найдено!")

            # Предобработка (is_phone_image=True для фото с телефона)
            processed_image = self.preprocessor.preprocess_single_image(
                image_path, is_phone_image=True
            )

            # Извлечение признаков
            features = self.feature_extractor.extract_all_features(processed_image)
            feature_vector = np.array(list(features.values())).reshape(1, -1)

            # Масштабирование
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Предсказание вероятностей
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]

            # Сопоставление вероятностей с именами классов
            class_names = self.label_encoder.classes_
            predictions = list(zip(class_names, probabilities))

            # Сортировка по убыванию вероятности
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Анализ особенностей почерка
            handwriting_analysis = self.analyze_handwriting_features(features)

            return {
                'top_matches': predictions[:top_k],
                'feature_analysis': handwriting_analysis,
                'raw_features': features
            }

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            raise

    def analyze_handwriting_features(self, features):
        """Анализ особенностей почерка"""
        analysis = {}

        # Анализ на основе извлеченных признаков
        if 'h_proj_mean' in features:
            density = features.get('h_proj_mean', 0)
            if density > 1000:
                analysis['плотность'] = "Густой почерк"
            elif density < 500:
                analysis['плотность'] = "Разреженный почерк"
            else:
                analysis['плотность'] = "Средняя плотность"

        if 'contour_count' in features:
            count = features.get('contour_count', 0)
            if count > 50:
                analysis['детализация'] = "Высокая детализация"
            elif count < 20:
                analysis['детализация'] = "Упрощенный почерк"
            else:
                analysis['детализация'] = "Средняя детализация"

        if 'mean_area' in features:
            area = features.get('mean_area', 0)
            if area > 300:
                analysis['размер'] = "Крупный почерк"
            elif area < 100:
                analysis['размер'] = "Мелкий почерк"
            else:
                analysis['размер'] = "Средний почерк"

        return analysis

    def generate_report(self, prediction_result, image_path=None):
        """Генерация красивого отчета"""
        top_matches = prediction_result['top_matches']
        analysis = prediction_result['feature_analysis']

        report = []
        report.append("=" * 60)
        report.append("🎯 АНАЛИЗ ПОЧЕРКА - РЕЗУЛЬТАТЫ")
        report.append("=" * 60)

        if image_path:
            report.append(f"📁 Анализируемое изображение: {os.path.basename(image_path)}")

        report.append(f"\n🏆 ТОП-{len(top_matches)} РЕЗУЛЬТАТОВ:")
        report.append("-" * 40)

        for i, (ruler, confidence) in enumerate(top_matches, 1):
            stars = "⭐" * min(i, 3) if i <= 3 else ""
            report.append(f"{i:2d}. {ruler:25} {confidence:6.2%} {stars}")

        report.append(f"\n📊 ОСОБЕННОСТИ ВАШЕГО ПОЧЕРКА:")
        report.append("-" * 40)
        for feature, description in analysis.items():
            report.append(f"  • {feature}: {description}")

        report.append(f"\n💡 ИНТЕРПРЕТАЦИЯ:")
        report.append("-" * 40)
        best_match, best_confidence = top_matches[0]

        if best_confidence > 0.7:
            report.append(f"Высокая вероятность совпадения с {best_match}")
        elif best_confidence > 0.4:
            report.append(f"Умеренная вероятность совпадения с {best_match}")
        else:
            report.append(f"Неоднозначный результат. Наиболее вероятно: {best_match}")

        report.append("=" * 60)

        return "\n".join(report)

    def visualize_prediction(self, image_path, prediction_result, save_path=None):
        """Визуализация предсказания"""
        try:
            top_matches = prediction_result['top_matches']

            # Создаем график
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Первый subplot: оригинальное изображение
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax1.imshow(img_rgb)
                    ax1.set_title('Анализируемое изображение', fontsize=14, fontweight='bold')
                    ax1.axis('off')
                else:
                    ax1.text(0.5, 0.5, 'Изображение не загружено',
                             ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                    ax1.axis('off')
            else:
                ax1.text(0.5, 0.5, 'Файл не найден',
                         ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                ax1.axis('off')

            # Второй subplot: вероятности
            rulers = [match[0] for match in top_matches]
            probabilities = [match[1] for match in top_matches]

            colors = plt.cm.YlOrRd(np.linspace(0.6, 0.9, len(rulers)))
            bars = ax2.barh(rulers, probabilities, color=colors)

            # Добавляем значения на столбцы
            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{prob:.2%}', ha='left', va='center', fontweight='bold')

            ax2.set_xlabel('Вероятность совпадения', fontsize=12)
            ax2.set_title('Сходство с почерком правителей', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Визуализация сохранена: {save_path}")

            plt.show()

        except Exception as e:
            print(f"❌ Ошибка при визуализации: {e}")


def test_with_sample_image():
    """Тестирование системы на примере"""
    try:
        # Инициализация предсказателя
        predictor = HandwritingPredictor()

        # Путь к тестовому изображению
        test_image_path = "test.jpg"  # Простое имя файла в текущей папке

        # Если файл не существует, используем первый доступный из датасета
        if not os.path.exists(test_image_path):
            print("📝 Тестовое изображение не найдено. Использую пример из датасета...")
            # Найдем первое доступное изображение
            data_path = "data/rulers"
            for ruler in os.listdir(data_path):
                ruler_path = os.path.join(data_path, ruler)
                if os.path.isdir(ruler_path):
                    for file in os.listdir(ruler_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            test_image_path = os.path.join(ruler_path, file)
                            print(f"📁 Использую: {test_image_path}")
                            break
                    if os.path.exists(test_image_path):
                        break

        if not os.path.exists(test_image_path):
            print("❌ Не найдено ни одного тестового изображения!")
            print("💡 Создайте файл test_handwriting.jpg в папке проекта или добавьте изображения в data/rulers/")
            return

        # Предсказание
        result = predictor.predict_handwriting(test_image_path, top_k=5)

        # Генерация отчета
        report = predictor.generate_report(result, test_image_path)
        print(report)

        # Визуализация
        predictor.visualize_prediction(test_image_path, result, "prediction_result.png")

        return result

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_sample_image()