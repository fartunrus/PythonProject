import cv2
import numpy as np
from skimage import feature, morphology


class HandwritingFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_basic_features(self, binary_image):
        """Извлечение базовых признаков"""
        features = {}

        # Проекционные профили
        horizontal_proj = np.sum(binary_image, axis=1)
        vertical_proj = np.sum(binary_image, axis=0)

        features['h_proj_mean'] = np.mean(horizontal_proj)
        features['h_proj_std'] = np.std(horizontal_proj)
        features['v_proj_mean'] = np.mean(vertical_proj)
        features['v_proj_std'] = np.std(vertical_proj)

        # Плотность текста
        features['text_density'] = np.sum(binary_image) / binary_image.size

        return features

    def extract_contour_features(self, binary_image):
        """Извлечение признаков из контуров"""
        contours, _ = cv2.findContours(binary_image.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = {}

        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

            features['contour_count'] = len(contours)
            features['mean_area'] = np.mean(areas)
            features['area_std'] = np.std(areas)
            features['mean_perimeter'] = np.mean(perimeters)

            # Форма контуров
            circularities = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularities.append(circularity)

            if circularities:
                features['mean_circularity'] = np.mean(circularities)
            else:
                features['mean_circularity'] = 0
        else:
            features['contour_count'] = 0
            features['mean_area'] = 0
            features['area_std'] = 0
            features['mean_perimeter'] = 0
            features['mean_circularity'] = 0

        return features

    def extract_texture_features(self, image):
        """Текстурные признаки с помощью LBP"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        # Простой LBP
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-8)

        features = {}
        for i, val in enumerate(lbp_hist):
            features[f'texture_{i}'] = val

        return features

    def extract_all_features(self, image):
        """Извлечение всех признаков из изображения"""
        try:
            print("  Извлечение признаков...")

            # Конвертация в binary
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)

            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            features = {}

            # Базовые признаки
            basic_features = self.extract_basic_features(binary)
            features.update(basic_features)

            # Признаки контуров
            contour_features = self.extract_contour_features(binary)
            features.update(contour_features)

            # Текстурные признаки
            texture_features = self.extract_texture_features(image)
            features.update(texture_features)

            self.feature_names = list(features.keys())
            print(f"  Извлечено {len(features)} признаков")
            return features

        except Exception as e:
            print(f"  Ошибка при извлечении признаков: {e}")
            raise