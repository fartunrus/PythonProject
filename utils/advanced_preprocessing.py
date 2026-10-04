import cv2
import numpy as np
from skimage import morphology, measure
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class AdvancedHandwritingPreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def robust_image_load(self, image_path):
        """Надежная загрузка изображения через PIL с обработкой ошибок"""
        try:
            if isinstance(image_path, str):
                # Пробуем загрузить через PIL (более надежно с путями)
                pil_image = Image.open(image_path)

                # Конвертируем в numpy array
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')

                image = np.array(pil_image)

                # Конвертируем BGR в RGB если нужно
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                return image
            else:
                # Если уже загружено изображение
                return image_path

        except Exception as e:
            print(f"  Ошибка загрузки изображения: {e}")
            return None

    def smart_resize_with_padding(self, image):
        """Умное изменение размера с сохранением пропорций"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # Вычисляем scaling factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize с сохранением пропорций
        resized = cv2.resize(image, (new_w, new_h))

        # Добавляем padding для достижения целевого размера
        delta_h = target_h - new_h
        delta_w = target_w - new_w

        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        # Добавляем белый padding
        if len(resized.shape) == 3:
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=255)

        return padded

    def extract_text_lines(self, image):
        """Сегментация текста на строки"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Проекционный профиль по горизонтали
        horizontal_projection = np.sum(binary, axis=1)

        # Находим строки через поиск пиков
        lines = []
        in_line = False
        line_start = 0

        threshold = np.mean(horizontal_projection) * 0.3

        for i, value in enumerate(horizontal_projection):
            if value > threshold and not in_line:
                in_line = True
                line_start = i
            elif value <= threshold and in_line:
                in_line = False
                line_end = i
                line_height = line_end - line_start
                if line_height > 10:  # Минимальная высота строки
                    lines.append((line_start, line_end))

        return lines

    def normalize_line_spacing(self, image, target_line_height=50):
        """Нормализация межстрочного расстояния"""
        lines = self.extract_text_lines(image)

        if len(lines) < 2:
            return image  # Нечего нормализовать

        # Вычисляем среднюю высоту строк
        line_heights = [end - start for start, end in lines]
        avg_line_height = np.mean(line_heights)

        if avg_line_height == 0:
            return image

        # Вычисляем scaling factor для нормализации высоты строк
        scale_factor = target_line_height / avg_line_height

        # Применяем масштабирование
        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        resized = cv2.resize(image, (new_w, new_h))

        return resized

    def remove_background_noise(self, image):
        """Удаление фонового шума и артефактов"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Adaptive threshold для лучшего выделения текста
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Морфологические операции для удаления шума
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def extract_writing_region(self, image):
        """Выделение основной области письма"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Объединяем все контуры текста
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])

        if len(all_points) == 0:
            return image

        # Находим bounding box всего текста
        x, y, w, h = cv2.boundingRect(all_points)

        # Добавляем padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Вырезаем область текста
        if len(image.shape) == 3:
            text_region = image[y:y + h, x:x + w]
        else:
            text_region = image[y:y + h, x:x + w]

        return text_region

    def enhance_contrast(self, image):
        """Улучшение контраста"""
        if len(image.shape) == 3:
            # Для цветных изображений
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Для grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

        return enhanced

    def advanced_preprocess(self, image_path, is_phone_image=False):
        """Расширенный пайплайн предобработки"""
        try:
            # Надежная загрузка изображения
            image = self.robust_image_load(image_path)

            if image is None:
                raise ValueError("Не удалось загрузить изображение")

            print(f"  Исходный размер: {image.shape}")

            # Удаление фонового шума
            cleaned = self.remove_background_noise(image)

            # Выделение области письма
            text_region = self.extract_writing_region(cleaned)

            # Для фото с телефона - удаление клетки
            if is_phone_image:
                # Упрощенное удаление клетки через размытие
                kernel = np.ones((5, 5), np.float32) / 25
                smoothed = cv2.filter2D(text_region, -1, kernel)
                text_region = cv2.addWeighted(text_region, 1.5, smoothed, -0.5, 0)

            # Нормализация межстрочного расстояния (если есть несколько строк)
            lines = self.extract_text_lines(text_region)
            if len(lines) >= 2:
                text_region = self.normalize_line_spacing(text_region)

            # Улучшение контраста
            enhanced = self.enhance_contrast(text_region)

            # Умное изменение размера
            normalized = self.smart_resize_with_padding(enhanced)

            # Финальная нормализация
            if len(normalized.shape) == 3:
                normalized = normalized.astype(np.float32) / 255.0
            else:
                normalized = normalized.astype(np.float32) / 255.0
                normalized = np.stack([normalized] * 3, axis=-1)  # Convert to 3-channel

            print(f"  Финальный размер: {normalized.shape}")

            return normalized

        except Exception as e:
            print(f"  Ошибка расширенной предобработки: {e}")
            # Fallback к простой предобработке
            try:
                from utils.preprocessing import HandwritingPreprocessor
                basic_preprocessor = HandwritingPreprocessor()
                return basic_preprocessor.preprocess_single_image(image_path, is_phone_image)
            except:
                # Создаем пустое изображение как fallback
                print("  Создание fallback изображения")
                return np.ones((512, 512, 3), dtype=np.float32) * 0.5