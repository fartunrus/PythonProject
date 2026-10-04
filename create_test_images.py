import os
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np


def create_realistic_handwriting(ruler_name, sample_num):
    """Создание более реалистичных изображений рукописного текста"""
    # Разные стили для разных правителей
    styles = {
        "Наполеон": {
            "slant": random.uniform(-15, -5),  # Левый наклон
            "pressure": random.uniform(2, 4),  # Сильный нажим
            "size": random.uniform(30, 40),  # Крупный почерк
            "spacing": random.uniform(15, 25)  # Широкие расстояния
        },
        "Никон": {
            "slant": random.uniform(5, 15),  # Правый наклон
            "pressure": random.uniform(1, 2),  # Слабый нажим
            "size": random.uniform(20, 30),  # Мелкий почерк
            "spacing": random.uniform(10, 15)  # Узкие расстояния
        },
        "Петр_Великий": {
            "slant": random.uniform(-5, 5),  # Почти вертикальный
            "pressure": random.uniform(2, 3),  # Средний нажим
            "size": random.uniform(25, 35),  # Средний размер
            "spacing": random.uniform(12, 20)  # Средние расстояния
        }
    }

    style = styles[ruler_name]

    # Создаем изображение
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Имитируем несколько строк текста
    y_position = 50
    for line_num in range(5):
        x_position = 50
        # Создаем "слова" из случайных линий
        for word_num in range(8):
            # Базовые параметры букв
            letter_height = style["size"]
            letter_width = letter_height * random.uniform(0.5, 0.8)

            # Рисуем "букву" как набор линий
            for stroke in range(3):
                start_x = x_position + random.uniform(0, letter_width)
                start_y = y_position + random.uniform(0, letter_height)
                end_x = start_x + random.uniform(-style["slant"], style["slant"])
                end_y = start_y + letter_height * random.uniform(0.8, 1.2)

                # Толщина линии в зависимости от нажима
                thickness = int(style["pressure"])
                draw.line([(start_x, start_y), (end_x, end_y)],
                          fill='black', width=thickness)

            x_position += letter_width + style["spacing"]

        y_position += style["size"] + style["spacing"] * 1.5

    # Добавляем немного шума для реалистичности
    img_array = np.array(img)
    noise = np.random.randint(0, 20, img_array.shape, dtype=np.uint8)
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    return img


def create_realistic_dataset():
    """Создание реалистичного датасета"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(project_root, "data", "rulers_realistic")

    print(f"Создание реалистичного датасета в: {base_path}")

    rulers = ["Наполеон", "Никон", "Петр_Великий"]

    for ruler in rulers:
        ruler_path = os.path.join(base_path, ruler)
        os.makedirs(ruler_path, exist_ok=True)

        print(f"Создание изображений для {ruler}")

        # Создаем больше изображений для каждого правителя
        for i in range(10):  # 10 изображений на класс
            img = create_realistic_handwriting(ruler, i)
            filename = os.path.join(ruler_path, f"realistic_{i}.jpg")
            img.save(filename, 'JPEG', quality=95)
            print(f"  Создано: {filename}")

    print("Реалистичный датасет создан!")


if __name__ == "__main__":
    create_realistic_dataset()