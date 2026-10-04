import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from sklearn.utils import compute_class_weight


class BalancedDataLoader:
    def __init__(self, target_samples_per_class=15):
        self.target_samples_per_class = target_samples_per_class

    def analyze_dataset(self, data_path):
        """Анализ распределения данных по классам"""
        class_stats = {}
        total_images = 0

        for ruler_name in os.listdir(data_path):
            ruler_path = os.path.join(data_path, ruler_name)
            if os.path.isdir(ruler_path):
                images = [f for f in os.listdir(ruler_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_stats[ruler_name] = len(images)
                total_images += len(images)

        print("Статистика датасета:")
        for ruler, count in class_stats.items():
            print(f"  {ruler}: {count} изображений")
        print(f"Всего изображений: {total_images}")

        return class_stats

    def augment_image(self, image, augmentation_count):
        """Аугментация изображения для балансировки классов"""
        augmented_images = []

        for i in range(augmentation_count):
            # Создаем копию для аугментации
            aug_img = image.copy()

            # Случайное вращение (-10 до +10 градусов)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                aug_img = aug_img.rotate(angle, resample=Image.BICUBIC, expand=False)

            # Случайное смещение
            if random.random() > 0.5:
                x_shift = random.randint(-5, 5)
                y_shift = random.randint(-5, 5)
                aug_img = aug_img.transform(aug_img.size, Image.AFFINE,
                                            (1, 0, x_shift, 0, 1, y_shift))

            # Случайное изменение контраста
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(aug_img)
                aug_img = enhancer.enhance(contrast_factor)

            # Случайное изменение яркости
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Brightness(aug_img)
                aug_img = enhancer.enhance(brightness_factor)

            # Случайное размытие
            if random.random() > 0.7:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))

            augmented_images.append(aug_img)

        return augmented_images

    def load_balanced_dataset(self, data_path, preprocessor, feature_extractor):
        """Загрузка данных с балансировкой классов"""
        class_stats = self.analyze_dataset(data_path)
        features_list = []
        labels_list = []

        print("\nБалансировка данных...")

        for ruler_name, image_count in class_stats.items():
            ruler_path = os.path.join(data_path, ruler_name)
            print(f"\nОбработка {ruler_name} ({image_count} исходных изображений)")

            # Загружаем оригинальные изображения
            original_processed = []
            successful_images = 0

            for image_file in os.listdir(ruler_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(ruler_path, image_file)
                    try:
                        print(f"  Загрузка {image_file}...")

                        # Загружаем через preprocessor (он сам обработает путь)
                        processed_image = preprocessor.preprocess_single_image(
                            image_path, is_phone_image=False
                        )
                        original_processed.append(processed_image)
                        successful_images += 1

                    except Exception as e:
                        print(f"  Ошибка загрузки {image_file}: {e}")
                        continue

            # Если ни одно изображение не загружено, пропускаем класс
            if successful_images == 0:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить ни одного изображения для {ruler_name}. Пропускаем.")
                continue

            print(f"  Успешно загружено: {successful_images} изображений")

            # Определяем сколько аугментаций нужно
            if successful_images < self.target_samples_per_class:
                augmentations_needed = self.target_samples_per_class - successful_images
                augmentations_per_image = max(1, augmentations_needed // successful_images)

                print(f"  Создание {augmentations_needed} аугментированных изображений "
                      f"({augmentations_per_image} на оригинал)")

                # Создаем аугментированные версии
                augmentation_count = 0
                for i, processed_image in enumerate(original_processed):
                    # Конвертируем обратно в PIL для аугментации
                    img_uint8 = (processed_image * 255).astype(np.uint8)
                    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
                        pil_img = Image.fromarray(img_uint8, 'RGB')
                    else:
                        # Если grayscale
                        pil_img = Image.fromarray(img_uint8)

                    # Создаем аугментированные версии
                    augmented_images = self.augment_image(pil_img, augmentations_per_image)

                    for aug_img in augmented_images:
                        # Конвертируем обратно в numpy и обрабатываем
                        aug_np = np.array(aug_img).astype(np.float32) / 255.0

                        # Извлекаем признаки из аугментированного изображения
                        features = feature_extractor.extract_all_features(aug_np)
                        features_list.append(list(features.values()))
                        labels_list.append(ruler_name)
                        augmentation_count += 1

                        if augmentation_count >= augmentations_needed:
                            break

                    if augmentation_count >= augmentations_needed:
                        break

            # Добавляем оригинальные изображения
            for processed_image in original_processed:
                features = feature_extractor.extract_all_features(processed_image)
                features_list.append(list(features.values()))
                labels_list.append(ruler_name)

            final_count = len([x for x in labels_list if x == ruler_name])
            print(f"  Итоговое количество: {final_count} изображений")

        if not features_list:
            raise ValueError("Не удалось загрузить ни одного изображения из датасета!")

        return np.array(features_list), np.array(labels_list)

    def compute_class_weights(self, labels):
        """Вычисление весов классов для компенсации дисбаланса"""
        from sklearn.utils.class_weight import compute_class_weight
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        return dict(zip(unique_labels, class_weights))