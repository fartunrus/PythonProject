import cv2
import numpy as np
from scipy import stats, fft, ndimage
from skimage import feature, measure, filters, morphology
from scipy.signal import find_peaks
import math


class AdvancedFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_content_invariant_features(self, image):
        """Признаки, инвариантные к содержанию текста"""
        features = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Статистики распределения черного/белого
        features.update(self.extract_spatial_distribution(binary))

        # 2. Текстурные признаки (не зависят от содержания)
        features.update(self.extract_advanced_texture(gray))

        # 3. Статистики контуров
        features.update(self.extract_contour_statistics(binary))

        # 4. Признаки на основе проекционных профилей
        features.update(self.extract_projection_features(binary))

        # 5. Спектральные признаки
        features.update(self.extract_spectral_features(binary))

        # 6. Геометрические признаки
        features.update(self.extract_geometric_features(binary))

        # 7. Ритмические признаки
        features.update(self.extract_rhythm_features(binary))

        return features

    def extract_spatial_distribution(self, binary):
        """Статистики пространственного распределения"""
        features = {}

        # Центр масс
        y_coords, x_coords = np.where(binary > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            features['center_x_norm'] = center_x / binary.shape[1]
            features['center_y_norm'] = center_y / binary.shape[0]

            # Моменты инерции
            features['inertia_x'] = np.std(x_coords) / binary.shape[1]
            features['inertia_y'] = np.std(y_coords) / binary.shape[0]
        else:
            features.update({'center_x_norm': 0.5, 'center_y_norm': 0.5,
                             'inertia_x': 0, 'inertia_y': 0})

        # Распределение по квадрантам
        h, w = binary.shape
        quadrants = [
            binary[:h // 2, :w // 2],  # верхний левый
            binary[:h // 2, w // 2:],  # верхний правый
            binary[h // 2:, :w // 2],  # нижний левый
            binary[h // 2:, w // 2:]  # нижний правый
        ]

        for i, quad in enumerate(quadrants):
            density = np.sum(quad) / (quad.size + 1e-8)  # избегаем деления на 0
            features[f'quadrant_{i}_density'] = density

        return features

    def extract_advanced_texture(self, gray_image):
        """Расширенные текстурные признаки"""
        features = {}

        # LBP с разными параметрами
        for radius in [1, 2, 3]:
            n_points = 8 * radius
            try:
                lbp = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
                lbp_hist = lbp_hist.astype(float)
                lbp_hist /= (lbp_hist.sum() + 1e-8)

                for j, val in enumerate(lbp_hist[:5]):  # Берем первые 5 бинов
                    features[f'lbp_r{radius}_b{j}'] = val
            except Exception as e:
                # Если LBP не работает, заполняем нулями
                for j in range(5):
                    features[f'lbp_r{radius}_b{j}'] = 0.0

        # Градиентные признаки
        try:
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            features['gradient_mean'] = np.mean(magnitude)
            features['gradient_std'] = np.std(magnitude)

            # Энтропия градиента с обработкой нулевых значений
            hist, _ = np.histogram(magnitude, bins=10)
            hist = hist[hist > 0]  # убираем нулевые бины
            if len(hist) > 0:
                features['gradient_entropy'] = stats.entropy(hist)
            else:
                features['gradient_entropy'] = 0
        except Exception as e:
            features.update({'gradient_mean': 0, 'gradient_std': 0, 'gradient_entropy': 0})

        return features

    def extract_contour_statistics(self, binary):
        """Статистики контуров"""
        contours, _ = cv2.findContours(binary.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = {}

        if contours:
            areas = []
            perimeters = []
            compactness = []
            eccentricities = []

            for contour in contours:
                if len(contour) >= 5:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)

                    if perimeter > 0:
                        areas.append(area)
                        perimeters.append(perimeter)
                        compactness.append(4 * np.pi * area / (perimeter * perimeter))

                    # Эксцентриситет
                    if len(contour) >= 5:
                        try:
                            ellipse = cv2.fitEllipse(contour)
                            (x, y), (MA, ma), angle = ellipse
                            if MA > 0:
                                eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2)
                                eccentricities.append(eccentricity)
                        except:
                            pass

            if areas:
                features.update({
                    'contour_area_mean': np.mean(areas),
                    'contour_area_std': np.std(areas) if len(areas) > 1 else 0,
                    'contour_perimeter_mean': np.mean(perimeters),
                    'compactness_mean': np.mean(compactness),
                    'eccentricity_mean': np.mean(eccentricities) if eccentricities else 0
                })
            else:
                features.update({
                    'contour_area_mean': 0, 'contour_area_std': 0,
                    'contour_perimeter_mean': 0, 'compactness_mean': 0,
                    'eccentricity_mean': 0
                })
        else:
            features.update({
                'contour_area_mean': 0, 'contour_area_std': 0,
                'contour_perimeter_mean': 0, 'compactness_mean': 0,
                'eccentricity_mean': 0
            })

        return features

    def extract_projection_features(self, binary):
        """Признаки на основе проекционных профилей"""
        features = {}

        # Горизонтальный проекционный профиль
        horizontal_proj = np.sum(binary, axis=1)
        vertical_proj = np.sum(binary, axis=0)

        # Нормализуем по длине
        if len(horizontal_proj) > 0 and np.max(horizontal_proj) > 0:
            horizontal_proj = horizontal_proj / np.max(horizontal_proj)
        if len(vertical_proj) > 0 and np.max(vertical_proj) > 0:
            vertical_proj = vertical_proj / np.max(vertical_proj)

        # Статистики профилей
        for proj, name in [(horizontal_proj, 'horizontal'), (vertical_proj, 'vertical')]:
            if len(proj) > 0:
                features[f'{name}_proj_mean'] = np.mean(proj)
                features[f'{name}_proj_std'] = np.std(proj) if len(proj) > 1 else 0

                # Асимметрия и эксцесс с обработкой особых случаев
                if len(proj) > 2:
                    try:
                        features[f'{name}_proj_skew'] = stats.skew(proj)
                        features[f'{name}_proj_kurtosis'] = stats.kurtosis(proj)
                    except:
                        features[f'{name}_proj_skew'] = 0
                        features[f'{name}_proj_kurtosis'] = 0
                else:
                    features[f'{name}_proj_skew'] = 0
                    features[f'{name}_proj_kurtosis'] = 0

                # Автокорреляция (периодичность)
                if len(proj) > 1:
                    autocorr = np.correlate(proj, proj, mode='full')
                    autocorr = autocorr[len(autocorr) // 2:]
                    if len(autocorr) > 1 and autocorr[0] > 0:
                        features[f'{name}_autocorr_decay'] = autocorr[1] / autocorr[0]
                    else:
                        features[f'{name}_autocorr_decay'] = 0
                else:
                    features[f'{name}_autocorr_decay'] = 0
            else:
                features.update({f'{name}_proj_mean': 0, f'{name}_proj_std': 0,
                                 f'{name}_proj_skew': 0, f'{name}_proj_kurtosis': 0,
                                 f'{name}_autocorr_decay': 0})

        return features

    def extract_spectral_features(self, binary):
        """Спектральные признаки (Фурье анализ)"""
        features = {}

        try:
            # 2D Фурье преобразование
            f_transform = fft.fft2(binary)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # Усредняем по углам (радиальный спектр)
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            y, x = np.indices(magnitude_spectrum.shape)
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            r = r.astype(int)

            tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
            rbin = np.bincount(r.ravel())
            radial_profile = tbin / (rbin + 1e-8)

            if len(radial_profile) > 5:
                # Энергия в низких и высоких частотах
                low_freq_energy = np.sum(radial_profile[:len(radial_profile) // 4])
                high_freq_energy = np.sum(radial_profile[3 * len(radial_profile) // 4:])
                total_energy = np.sum(radial_profile)

                if total_energy > 0:
                    features['low_freq_ratio'] = low_freq_energy / total_energy
                    features['high_freq_ratio'] = high_freq_energy / total_energy
                else:
                    features.update({'low_freq_ratio': 0, 'high_freq_ratio': 0})
            else:
                features.update({'low_freq_ratio': 0, 'high_freq_ratio': 0})
        except Exception as e:
            features.update({'low_freq_ratio': 0, 'high_freq_ratio': 0})

        return features

    def extract_geometric_features(self, binary):
        """Геометрические признаки почерка"""
        features = {}

        try:
            # Ориентация и выравнивание
            moments = cv2.moments(binary.astype(np.uint8))
            if moments['mu02'] > 0:
                # Угол ориентации
                angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
                features['orientation_angle'] = angle

                # Эксцентриситет
                lambda1 = moments['mu20'] + moments['mu02'] + np.sqrt(
                    (moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2)
                lambda2 = moments['mu20'] + moments['mu02'] - np.sqrt(
                    (moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2)
                if lambda1 > 0:
                    features['eccentricity'] = np.sqrt(1 - lambda2 / lambda1)
                else:
                    features['eccentricity'] = 0
            else:
                features.update({'orientation_angle': 0, 'eccentricity': 0})
        except:
            features.update({'orientation_angle': 0, 'eccentricity': 0})

        # Плотность и распределение
        try:
            labeled_image = measure.label(binary)
            regions = measure.regionprops(labeled_image)

            if regions:
                areas = [region.area for region in regions]
                features['connected_components'] = len(regions)
                features['avg_component_area'] = np.mean(areas)
                features['component_area_std'] = np.std(areas) if len(areas) > 1 else 0
            else:
                features.update({'connected_components': 0, 'avg_component_area': 0, 'component_area_std': 0})
        except:
            features.update({'connected_components': 0, 'avg_component_area': 0, 'component_area_std': 0})

        return features

    def extract_rhythm_features(self, binary):
        """Анализ ритма и периодичности почерка"""
        features = {}

        try:
            # Горизонтальный проекционный профиль - убеждаемся, что это 1D массив
            horizontal_proj = np.sum(binary, axis=1)

            # Преобразуем к 1D массиву если нужно
            if horizontal_proj.ndim > 1:
                horizontal_proj = horizontal_proj.flatten()

            if len(horizontal_proj) > 10:
                # Анализ Фурье для выявления периодичности
                fft_result = np.abs(fft.fft(horizontal_proj))
                fft_result = fft_result[:len(fft_result) // 2]  # Берем первую половину

                # Пики в спектре (используем find_peaks из scipy.signal)
                if len(fft_result) > 1:
                    try:
                        peaks, _ = find_peaks(fft_result, height=np.mean(fft_result))
                        features['rhythm_peak_count'] = len(peaks)

                        if len(peaks) > 0:
                            features['rhythm_dominant_freq'] = peaks[0] / len(horizontal_proj)
                            features['rhythm_peak_strength'] = np.mean(fft_result[peaks])
                        else:
                            features.update(
                                {'rhythm_peak_count': 0, 'rhythm_dominant_freq': 0, 'rhythm_peak_strength': 0})
                    except Exception as e:
                        features.update({'rhythm_peak_count': 0, 'rhythm_dominant_freq': 0, 'rhythm_peak_strength': 0})
                else:
                    features.update({'rhythm_peak_count': 0, 'rhythm_dominant_freq': 0, 'rhythm_peak_strength': 0})
            else:
                features.update({'rhythm_peak_count': 0, 'rhythm_dominant_freq': 0, 'rhythm_peak_strength': 0})
        except Exception as e:
            features.update({'rhythm_peak_count': 0, 'rhythm_dominant_freq': 0, 'rhythm_peak_strength': 0})

        return features

    def extract_all_features(self, image):
        """Извлечение всех улучшенных признаков"""
        try:
            print("  Извлечение улучшенных признаков...")

            features = self.extract_content_invariant_features(image)

            # Заменяем любые NaN значения на 0
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0

            self.feature_names = list(features.keys())
            print(f"  Извлечено {len(features)} улучшенных признаков")
            return features

        except Exception as e:
            print(f"  Ошибка при извлечении улучшенных признаков: {e}")
            # Возвращаем пустые признаки как последнее средство
            return {name: 0 for name in self.feature_names} if self.feature_names else {}