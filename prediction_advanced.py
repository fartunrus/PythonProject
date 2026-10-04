import os
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils.advanced_preprocessing import AdvancedHandwritingPreprocessor
from utils.advanced_feature_extraction import AdvancedFeatureExtractor


class AdvancedHandwritingPredictor:
    def __init__(self, models_path='models_advanced'):
        self.models_path = models_path
        self.preprocessor = AdvancedHandwritingPreprocessor()
        self.load_models()

    def load_models(self):
        """Загрузка обученных моделей и компонентов"""
        try:
            print("Загрузка улучшенных моделей...")

            # Проверяем существование папки models_advanced
            if not os.path.exists(self.models_path):
                raise FileNotFoundError(
                    f"Папка {self.models_path} не найдена! Сначала выполните обучение training_advanced.py.")

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

            # Загрузка информации о датасете
            dataset_info_path = os.path.join(self.models_path, 'dataset_info.pkl')
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'rb') as f:
                    self.dataset_info = pickle.load(f)
                print("✅ Информация о датасете загружена")
            else:
                self.dataset_info = None
                print("⚠️  Информация о датасете не найдена")

            print("✅ Улучшенные модели и компоненты успешно загружены")
            print(f"✅ Классы в модели: {list(self.label_encoder.classes_)}")
            print(f"✅ Количество признаков: {len(self.feature_extractor.feature_names)}")

        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            print("\n💡 Решение: Запустите training_advanced.py для обучения улучшенных моделей")
            raise

    def predict_handwriting(self, image_path, top_k=5):
        """
        Предсказание схожести почерка с правителями с использованием улучшенных признаков
        """
        try:
            print(f"\n🔍 Анализ изображения: {os.path.basename(image_path)}")

            # Проверяем существование файла
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Изображение {image_path} не найдено!")

            # Улучшенная предобработка (is_phone_image=True для фото с телефона)
            processed_image = self.preprocessor.advanced_preprocess(
                image_path, is_phone_image=True
            )

            # Извлечение улучшенных признаков
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

            # Анализ особенностей почерка на основе улучшенных признаков
            handwriting_analysis = self.analyze_advanced_handwriting_features(features)

            return {
                'top_matches': predictions[:top_k],
                'feature_analysis': handwriting_analysis,
                'raw_features': features
            }

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {e}")
            raise

    def analyze_advanced_handwriting_features(self, features):
        """Анализ особенностей почерка на основе улучшенных признаков"""
        analysis = {}

        # Анализ пространственного распределения
        center_x = features.get('center_x_norm', 0.5)
        center_y = features.get('center_y_norm', 0.5)

        if center_x < 0.4:
            analysis['выравнивание'] = "Смещение влево"
        elif center_x > 0.6:
            analysis['выравнивание'] = "Смещение вправо"
        else:
            analysis['выравнивание'] = "Центрированный почерк"

        if center_y < 0.4:
            analysis['расположение'] = "Смещение вверх"
        elif center_y > 0.6:
            analysis['расположение'] = "Смещение вниз"
        else:
            analysis['расположение'] = "Центрированное расположение"

        # Анализ плотности по квадрантам
        quadrant_densities = []
        for i in range(4):
            key = f'quadrant_{i}_density'
            if key in features:
                quadrant_densities.append(features[key])

        if quadrant_densities:
            max_quadrant = np.argmax(quadrant_densities)
            quadrant_names = ['верхний левый', 'верхний правый', 'нижний левый', 'нижний правый']
            analysis['активная зона'] = f"Наибольшая активность в {quadrant_names[max_quadrant]} квадранте"

        # Анализ текстурных характеристик
        gradient_mean = features.get('gradient_mean', 0)
        if gradient_mean > 50:
            analysis['резкость'] = "Резкие, четкие штрихи"
        elif gradient_mean < 20:
            analysis['резкость'] = "Мягкие, плавные штрихи"
        else:
            analysis['резкость'] = "Средняя резкость штрихов"

        # Анализ геометрических характеристик
        orientation = features.get('orientation_angle', 0)
        if orientation > 0.1:
            analysis['наклон'] = "Правый наклон"
        elif orientation < -0.1:
            analysis['наклон'] = "Левый наклон"
        else:
            analysis['наклон'] = "Вертикальный почерк"

        eccentricity = features.get('eccentricity', 0)
        if eccentricity > 0.8:
            analysis['форма'] = "Вытянутый, угловатый почерк"
        elif eccentricity < 0.3:
            analysis['форма'] = "Округлый, плавный почерк"
        else:
            analysis['форма'] = "Сбалансированная форма"

        # Анализ распределения компонентов
        connected_components = features.get('connected_components', 0)
        if connected_components > 100:
            analysis['связность'] = "Раздельный почерк"
        elif connected_components < 30:
            analysis['связность'] = "Связный почерк"
        else:
            analysis['связность'] = "Смешанный почерк"

        # Анализ спектральных характеристик
        low_freq_ratio = features.get('low_freq_ratio', 0)
        if low_freq_ratio > 0.7:
            analysis['структура'] = "Регулярный, предсказуемый почерк"
        elif low_freq_ratio < 0.3:
            analysis['структура'] = "Сложный, изменчивый почерк"
        else:
            analysis['структура'] = "Сбалансированная структура"

        return analysis

    def generate_detailed_report(self, prediction_result, image_path=None):
        """Генерация детального отчета с улучшенной информацией"""
        top_matches = prediction_result['top_matches']
        analysis = prediction_result['feature_analysis']

        report = []
        report.append("=" * 70)
        report.append("🎯 УЛУЧШЕННЫЙ АНАЛИЗ ПОЧЕРКА - РЕЗУЛЬТАТЫ")
        report.append("=" * 70)

        if image_path:
            report.append(f"📁 Анализируемое изображение: {os.path.basename(image_path)}")

        report.append(f"\n🏆 ТОП-{len(top_matches)} СОВПАДЕНИЙ:")
        report.append("-" * 50)

        for i, (ruler, confidence) in enumerate(top_matches, 1):
            # Разные эмодзи для разных позиций
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔸"

            # Цветовые индикаторы уверенности
            if confidence > 0.7:
                confidence_indicator = "🟢 ВЫСОКАЯ"
            elif confidence > 0.4:
                confidence_indicator = "🟡 СРЕДНЯЯ"
            else:
                confidence_indicator = "🔴 НИЗКАЯ"

            report.append(f"{emoji} {i:2d}. {ruler:25} {confidence:6.2%} ({confidence_indicator})")

        report.append(f"\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ ОСОБЕННОСТЕЙ ПОЧЕРКА:")
        report.append("-" * 50)

        # Группируем анализ по категориям
        categories = {
            'Пространственные характеристики': ['выравнивание', 'расположение', 'активная зона'],
            'Геометрические свойства': ['наклон', 'форма', 'связность'],
            'Текстурные особенности': ['резкость', 'структура']
        }

        for category, features_list in categories.items():
            report.append(f"\n{category}:")
            for feature in features_list:
                if feature in analysis:
                    report.append(f"  • {feature.replace('_', ' ').title()}: {analysis[feature]}")

        report.append(f"\n💡 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        report.append("-" * 50)

        best_match, best_confidence = top_matches[0]
        second_match, second_confidence = top_matches[1] if len(top_matches) > 1 else (None, 0)

        if best_confidence > 0.7:
            report.append(f"✅ ВЫСОКАЯ СХОДИМОСТЬ: Ваш почерк имеет значительное сходство с {best_match}")
            if second_confidence > 0.3:
                report.append(f"   Также наблюдается сходство с {second_match}")
        elif best_confidence > 0.5:
            report.append(f"⚠️  УМЕРЕННАЯ СХОДИМОСТЬ: Ваш почерк имеет некоторые общие черты с {best_match}")
            if second_confidence > 0.3:
                report.append(f"   Дополнительное сходство с {second_match}")
        elif best_confidence > 0.3:
            report.append(f"🔍 СЛАБАЯ СХОДИМОСТЬ: Наибольшее сходство с {best_match}, но результат неоднозначен")
            report.append(f"   Рекомендуется предоставить больше текста для анализа")
        else:
            report.append(f"❓ НЕОПРЕДЕЛЕННЫЙ РЕЗУЛЬТАТ: Все совпадения имеют низкую уверенность")
            report.append(f"   Это может быть связано с уникальными особенностями вашего почерка")

        # Добавляем информацию о модели
        if self.dataset_info:
            report.append(f"\n📈 ИНФОРМАЦИЯ О МОДЕЛИ:")
            report.append("-" * 50)
            report.append(f"   • Обучена на {self.dataset_info['num_samples']} образцах")
            report.append(f"   • Анализирует {self.dataset_info['num_features']} признаков почерка")
            report.append(f"   • Сравнивает с {self.dataset_info['num_classes']} историческими личностями")

        report.append("=" * 70)

        return "\n".join(report)

    def visualize_advanced_prediction(self, image_path, prediction_result, save_path=None):
        """Улучшенная визуализация предсказания"""
        try:
            top_matches = prediction_result['top_matches']
            analysis = prediction_result['feature_analysis']

            # Создаем график с 3 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle('УЛУЧШЕННЫЙ АНАЛИЗ ПОЧЕРКА', fontsize=16, fontweight='bold')

            # 1. Оригинальное изображение
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

            # 2. Вероятности совпадения
            rulers = [match[0] for match in top_matches]
            probabilities = [match[1] for match in top_matches]

            colors = []
            for prob in probabilities:
                if prob > 0.7:
                    colors.append('#2ecc71')  # Зеленый для высокой уверенности
                elif prob > 0.4:
                    colors.append('#f39c12')  # Оранжевый для средней
                else:
                    colors.append('#e74c3c')  # Красный для низкой

            bars = ax2.barh(rulers, probabilities, color=colors)

            # Добавляем значения на столбцы
            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                color = 'white' if prob > 0.5 else 'black'
                ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{prob:.2%}', ha='left', va='center',
                         fontweight='bold', color=color, fontsize=10)

            ax2.set_xlabel('Вероятность совпадения', fontsize=12)
            ax2.set_title('Сходство с почерком правителей', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.grid(axis='x', alpha=0.3)

            # 3. Анализ особенностей почерка
            features_to_show = {
                'выравнивание': analysis.get('выравнивание', 'Н/Д'),
                'наклон': analysis.get('наклон', 'Н/Д'),
                'форма': analysis.get('форма', 'Н/Д'),
                'резкость': analysis.get('резкость', 'Н/Д'),
                'связность': analysis.get('связность', 'Н/Д')
            }

            y_pos = np.arange(len(features_to_show))
            ax3.barh(y_pos, [1] * len(features_to_show), color='lightblue', alpha=0.6)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features_to_show.keys())
            ax3.set_xlim(0, 1)
            ax3.axis('off')  # Убираем оси, так как это качественные данные

            # Добавляем текстовые описания
            for i, (feature, description) in enumerate(features_to_show.items()):
                ax3.text(0.05, i, description, va='center', ha='left', fontsize=10)

            ax3.set_title('Характеристики почерка', fontsize=14, fontweight='bold')

            # 4. Уверенность предсказания
            confidence = top_matches[0][1] if top_matches else 0
            confidence_levels = ['Низкая', 'Средняя', 'Высокая']
            confidence_values = [0.3, 0.6, 1.0]
            confidence_colors = ['#e74c3c', '#f39c12', '#2ecc71']

            current_level = next((i for i, val in enumerate(confidence_values) if confidence <= val), 2)

            ax4.barh(confidence_levels, confidence_values, color=confidence_colors, alpha=0.3)
            ax4.barh(confidence_levels[:current_level + 1], confidence_values[:current_level + 1],
                     color=confidence_colors[:current_level + 1])

            # Отмечаем текущую уверенность
            ax4.axvline(x=confidence, color='red', linestyle='--', linewidth=2,
                        label=f'Текущая уверенность: {confidence:.2%}')
            ax4.legend()

            ax4.set_xlabel('Уровень уверенности', fontsize=12)
            ax4.set_title('Оценка достоверности результата', fontsize=14, fontweight='bold')
            ax4.set_xlim(0, 1)
            ax4.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"📊 Визуализация сохранена: {save_path}")

            plt.show()

        except Exception as e:
            print(f"❌ Ошибка при визуализации: {e}")


def test_advanced_prediction():
    """Тестирование улучшенной системы на примере"""
    try:
        # Инициализация улучшенного предсказателя
        predictor = AdvancedHandwritingPredictor()

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
            print("💡 Создайте файл test_handwriting.jpg в папке проекта")
            print("   Или добавьте изображения в data/rulers/")
            return

        # Предсказание с улучшенной моделью
        result = predictor.predict_handwriting(test_image_path, top_k=5)

        # Генерация детального отчета
        report = predictor.generate_detailed_report(result, test_image_path)
        print(report)

        # Улучшенная визуализация
        predictor.visualize_advanced_prediction(test_image_path, result, "advanced_prediction_result.png")

        return result

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_advanced_prediction()