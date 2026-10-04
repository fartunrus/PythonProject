import cv2
import numpy as np
from skimage import filters, morphology, measure
import os
from PIL import Image


class HandwritingPreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def load_image(self, image_path):
        """Загрузка изображения с проверкой ошибок"""
        try:
            if isinstance(image_path, str):
                # Пробуем загрузить через OpenCV
                image = cv2.imread(image_path)

                if image is None:
                    # Если OpenCV не смог, пробуем через PIL
                    try:
                        pil_image = Image.open(image_path)
                        image = np.array(pil_image)
                        # Конвертируем RGB в BGR для OpenCV
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        elif len(image.shape) == 3 and image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    except Exception as pil_error:
                        raise ValueError(f"Не удалось загрузить изображение через PIL: {pil_error}")
            else:
                # Если передано уже загруженное изображение
                image = image_path

            if image is None:
                raise ValueError("Изображение не загружено (None)")

            return image

        except Exception as e:
            raise ValueError(f"Ошибка загрузки изображения {image_path}: {e}")

    def remove_grid_lines(self, image):
        """
        Упрощенное удаление клетки тетради
        """
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Простое размытие для удаления мелких деталей (клетки)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Повышение контраста текста
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def deskew_image(self, image):
        """
        Упрощенное выравнивание наклона текста
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Находим основной контур текста
        main_contour = max(contours, key=cv2.contourArea)

        # Получаем угол поворота
        rect = cv2.minAreaRect(main_contour)
        angle = rect[2]

        # Корректируем угол
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Поворачиваем изображение
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def enhance_contrast(self, image):
        """Упрощенное улучшение контраста"""
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

    def preprocess_single_image(self, image_path, is_phone_image=False):
        """
        Полный пайплайн предобработки одного изображения
        """
        try:
            # Безопасное получение имени файла для логов
            image_name = "unknown"
            if isinstance(image_path, str):
                image_name = os.path.basename(image_path)
                print(f"Обработка изображения: {image_name}")
            else:
                print("Обработка numpy массива")
                image_name = "numpy_array"

            image = self.load_image(image_path)

            if image is None:
                raise ValueError("Изображение не загружено")

            print(f"  Размер исходного изображения: {image.shape}")

            # Для фото с телефона - удаляем клетку и выравниваем
            if is_phone_image:
                print("  Удаление клетки...")
                image = self.remove_grid_lines(image)
                print("  Выравнивание...")
                image = self.deskew_image(image)

            print("  Улучшение контраста...")
            image = self.enhance_contrast(image)

            print("  Изменение размера...")
            image = cv2.resize(image, self.target_size)

            # Нормализация
            image = image.astype(np.float32) / 255.0

            print(f"  Обработка завершена. Финалный размер: {image.shape}")
            return image

        except Exception as e:
            print(f"  Ошибка при обработке {image_name}: {e}")
            raise

    def extract_text_regions(self, image):
        """
        Выделение регионов с текстом
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Бинаризация
        _, binary = cv2.threshold((gray * 255).astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Фильтр по размеру
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append((x, y, w, h))

        return text_regions