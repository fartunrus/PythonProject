import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
from sklearn.utils import compute_class_weight
from collections import Counter
import cv2


class EnhancedBalancedDataLoader:
    def __init__(self, target_samples_per_class=15):
        self.target_samples_per_class = target_samples_per_class
        self.augmentation_strategies = self._get_augmentation_strategies()
    def robust_load_image(self, image_path, preprocessor):
        """Надежная загрузка изображения с несколькими попытками"""
        try:
            # Пробуем загрузить через PIL
            pil_image = Image.open(image_path)
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')

            # Конвертируем в numpy array для обработки
            image_np = np.array(pil_image)

            # Обрабатываем через препроцессор
            processed_image = preprocessor.advanced_preprocess(image_np, is_phone_image=False)
            return processed_image

        except Exception as e:
            print(f"  Ошибка загрузки {os.path.basename(image_path)}: {e}")
            return None

    def _apply_augmentation(self, original_processed, ruler_name, preprocessor, feature_extractor,
                          features_list, labels_list, strategy, aug_per_image):
        """Применение аугментации к набору изображений"""
        augmentation_count = 0
        total_augmentations_needed = aug_per_image * len(original_processed)

        for i, processed_image in enumerate(original_processed):
            # Конвертируем обратно в PIL для аугментации
            img_uint8 = (processed_image * 255).astype(np.uint8)
            if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
                pil_img = Image.fromarray(img_uint8, 'RGB')
            else:
                pil_img = Image.fromarray(img_uint8)

            # Создаем аугментированные версии
            augmented_images = self.advanced_augment_image(pil_img, aug_per_image, strategy)

            for aug_img in augmented_images:
                if augmentation_count >= total_augmentations_needed:
                    break

                # Конвертируем обратно в numpy и обрабатываем
                aug_np = np.array(aug_img).astype(np.float32) / 255.0

                # Извлекаем признаки
                features = feature_extractor.extract_all_features(aug_np)
                if features and len(features) > 0:
                    features_list.append(list(features.values()))
                    labels_list.append(ruler_name)
                    augmentation_count += 1

            if augmentation_count >= total_augmentations_needed:
                break

        print(f"  Создано {augmentation_count} аугментированных изображений")
    def _get_augmentation_strategies(self):
        """Разные стратегии аугментации для разных типов почерка"""
        return {
            'aggressive': {
                'rotation_range': (-15, 15),
                'shift_range': (-10, 10),
                'contrast_range': (0.7, 1.5),
                'brightness_range': (0.7, 1.5)
            },
            'moderate': {
                'rotation_range': (-8, 8),
                'shift_range': (-5, 5),
                'contrast_range': (0.8, 1.3),
                'brightness_range': (0.8, 1.3)
            },
            'light': {
                'rotation_range': (-5, 5),
                'shift_range': (-3, 3),
                'contrast_range': (0.9, 1.1),
                'brightness_range': (0.9, 1.1)
            }
        }

    def analyze_dataset_balance(self, data_path):
        """Детальный анализ баланса датасета"""
        class_stats = {}
        for ruler_name in os.listdir(data_path):
            ruler_path = os.path.join(data_path, ruler_name)
            if os.path.isdir(ruler_path):
                images = [f for f in os.listdir(ruler_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_stats[ruler_name] = len(images)

        # Анализ дисбаланса
        counts = list(class_stats.values())
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')

        print("\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ БАЛАНСА:")
        print(f"   Коэффициент дисбаланса: {imbalance_ratio:.2f}")
        print(f"   Минимальное количество: {min(counts)}")
        print(f"   Максимальное количество: {max(counts)}")

        return class_stats, imbalance_ratio

    def smart_augmentation_strategy(self, original_count, target_count):
        """Умная стратегия аугментации в зависимости от дисбаланса"""
        needed = target_count - original_count
        if needed <= 0:
            return 'light', 0

        ratio = needed / original_count

        if ratio > 3:
            strategy = 'aggressive'
        elif ratio > 1.5:
            strategy = 'moderate'
        else:
            strategy = 'light'

        augmentations_per_image = max(1, needed // original_count)
        return strategy, augmentations_per_image

    def advanced_augment_image(self, image, augmentation_count, strategy_name='moderate'):
        """Улучшенная аугментация с разными стратегиями"""
        strategy = self.augmentation_strategies[strategy_name]
        augmented_images = []

        for i in range(augmentation_count):
            aug_img = image.copy()

            # Случайное вращение
            angle = random.uniform(*strategy['rotation_range'])
            aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

            # Случайное смещение
            x_shift = random.randint(*strategy['shift_range'])
            y_shift = random.randint(*strategy['shift_range'])
            aug_img = aug_img.transform(aug_img.size, Image.AFFINE,
                                        (1, 0, x_shift, 0, 1, y_shift), fillcolor=255)

            # Изменение контраста
            contrast_factor = random.uniform(*strategy['contrast_range'])
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(contrast_factor)

            # Изменение яркости
            brightness_factor = random.uniform(*strategy['brightness_range'])
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(brightness_factor)

            # Эластичные деформации (опционально)
            if random.random() > 0.7 and strategy_name == 'aggressive':
                aug_img = self.elastic_deformation(aug_img)

            # Добавление шума
            if random.random() > 0.6:
                aug_img = self.add_gaussian_noise(aug_img)

            augmented_images.append(aug_img)

        return augmented_images

    def elastic_deformation(self, image, alpha=20, sigma=5):
        """Эластичная деформация для имитации естественных вариаций почерка"""
        try:
            image_np = np.array(image)
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            shape = image_np.shape
            dx = np.random.uniform(-1, 1, shape) * alpha
            dy = np.random.uniform(-1, 1, shape) * alpha

            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            deformed = cv2.remap(image_np, indices[1].astype(np.float32),
                                 indices[0].astype(np.float32),
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)

            return Image.fromarray(deformed)
        except:
            return image  # Fallback

    def add_gaussian_noise(self, image, mean=0, sigma=5):
        """Добавление гауссовского шума"""
        image_np = np.array(image)
        noise = np.random.normal(mean, sigma, image_np.shape).astype(np.uint8)
        noisy_image = np.clip(image_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def load_balanced_dataset(self, data_path, preprocessor, feature_extractor):
        """Улучшенная загрузка с умной балансировкой"""
        class_stats, imbalance_ratio = self.analyze_dataset_balance(data_path)
        features_list = []
        labels_list = []

        print(f"\n🎯 ПРИМЕНЕНИЕ УМНОЙ БАЛАНСИРОВКИ...")

        for ruler_name, image_count in class_stats.items():
            ruler_path = os.path.join(data_path, ruler_name)

            if not os.path.isdir(ruler_path) or image_count == 0:
                continue

            print(f"\n📁 Обработка {ruler_name} ({image_count} изображений)")

            # Определяем стратегию аугментации
            strategy, aug_per_image = self.smart_augmentation_strategy(
                image_count, self.target_samples_per_class
            )

            # Загрузка и аугментация
            self._process_ruler_class(
                ruler_path, ruler_name, preprocessor, feature_extractor,
                features_list, labels_list, strategy, aug_per_image
            )

        if not features_list:
            raise ValueError("❌ Не удалось загрузить ни одного изображения!")

        return np.array(features_list), np.array(labels_list)

    def _process_ruler_class(self, ruler_path, ruler_name, preprocessor,
                             feature_extractor, features_list, labels_list,
                             strategy, aug_per_image):
        """Обработка одного класса правителя"""
        original_processed = []

        # Загрузка оригинальных изображений
        for image_file in os.listdir(ruler_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(ruler_path, image_file)
            processed_image = self.robust_load_image(image_path, preprocessor)

            if processed_image is not None:
                original_processed.append(processed_image)

        if not original_processed:
            print(f"  ⚠️  Пропуск {ruler_name} - нет загруженных изображений")
            return

        print(f"  ✅ Успешно загружено: {len(original_processed)} изображений")

        # Применяем аугментацию если нужно
        if aug_per_image > 0:
            print(f"  🔄 Аугментация ({strategy}): {aug_per_image} на изображение")
            self._apply_augmentation(original_processed, ruler_name, preprocessor,
                                     feature_extractor, features_list, labels_list,
                                     strategy, aug_per_image)

        # Добавляем оригинальные изображения
        for processed_image in original_processed:
            features = feature_extractor.extract_all_features(processed_image)
            if features and len(features) > 0:
                features_list.append(list(features.values()))
                labels_list.append(ruler_name)

        final_count = len([x for x in labels_list if x == ruler_name])
        print(f"  📊 Итоговое количество: {final_count} изображений")