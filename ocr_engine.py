# ocr_engine.py
import cv2
import numpy as np
import pytesseract
import requests
from io import BytesIO
from PIL import Image
import re

def clean_image(image_bytes):
    """Улучшает картинку для распознавания цифр."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None
    
    # 1. Переводим в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Увеличиваем контраст (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 3. Пороговая обработка (делаем текст черным на белом)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Немного увеличиваем
    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return thresh

def extract_price_from_url(url):
    """Скачивает и распознает цену."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None, "Ошибка скачивания"
            
        processed_img = clean_image(response.content)
        if processed_img is None:
            return None, "Ошибка чтения фото"

        # Tesseract настроен только на цифры
        # --psm 8 = распознавать одно слово (цену)
        # -c tessedit_char_whitelist=0123456789 = игнорировать всё кроме цифр
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Оставляем только цифры
        digits = re.sub(r'\D', '', text)
        
        if not digits:
            return None, "Цифры не найдены (возможно, плохое фото)"
        
        return int(digits), None
    except Exception as e:
        return None, str(e)
