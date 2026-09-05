# ============================================================
#   OCR-ДВИЖОК — автоматическое распознавание цифр на чеке.
#
#   Идея та же, что в присланном ocr_engine.py: Tesseract с белым
#   списком "только цифры" (--psm 8 = одно "слово"). Отличия от
#   исходника:
#
#   1. extract_digits_from_array(img) — принимает УЖЕ обработанную
#      картинку (результат filters_engine.apply_recipe), а не только
#      сырое фото. Это позволяет проверять OCR-ом любой рецепт из
#      таблицы filter_presets, а не только фиксированный clean_image
#      ниже — то есть "действительно ли этот фильтр помогает
#      распознаванию", а не только "на глаз читается".
#
#   2. extract_price_from_bgr(bgr) — старое поведение (свой
#      фиксированный clean_image), но принимает уже декодированный
#      numpy-массив, а не скачивает сам — переиспользует общий
#      requests+кэш из main.py вместо повторного скачивания.
#
#   3. extract_price_from_url(url) оставлена как есть (скачивает сама) —
#      удобно для разового локального запуска вне сервера/кэша main.py,
#      например точечно на CSV-выборке.
# ============================================================
import re

import numpy as np
import cv2
import pytesseract

# --oem 3       = движок по умолчанию (LSTM)
# --psm 8       = распознавать как одно "слово" (цена)
# whitelist     = игнорировать всё, кроме цифр
TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'


def clean_image(bgr: np.ndarray):
    """Фиксированная очистка для голого запуска без привязки к рецептам
    filters_engine (тот же алгоритм, что был в исходном ocr_engine.py)."""
    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return thresh


def extract_digits_from_array(img: np.ndarray):
    """Принимает УЖЕ обработанную картинку (серую/бинарную/цветную —
    не важно, любой выход apply_recipe или clean_image) и распознаёт
    цифры. Возвращает (digits_str_or_None, error_or_None)."""
    if img is None:
        return None, "пустое изображение"
    try:
        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        digits = re.sub(r'\D', '', text)
        if not digits:
            return None, "цифры не найдены (возможно, плохое фото)"
        return digits, None
    except pytesseract.TesseractNotFoundError:
        return None, ("Tesseract не установлен на сервере — нужен системный пакет "
                       "tesseract-ocr (см. Dockerfile, на Render 'Native' environment "
                       "это не установить через apt-get)")
    except Exception as e:
        return None, str(e)


def extract_price_from_bgr(bgr: np.ndarray):
    """Старое поведение исходного скрипта: своя фиксированная очистка +
    распознавание. Принимает уже декодированный BGR-массив (без
    повторного скачивания). Возвращает (int_or_None, error_or_None)."""
    processed = clean_image(bgr)
    digits, err = extract_digits_from_array(processed)
    if err:
        return None, err
    return int(digits), None


def extract_price_from_url(url: str):
    """Полностью автономная версия — сама качает фото. Для разовых
    локальных прогонов по CSV/списку URL вне сервера (без общего
    кэша и Supabase), как в исходном скрипте."""
    import requests

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None, "Ошибка скачивания"

        nparr = np.frombuffer(response.content, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "Ошибка чтения фото"

        return extract_price_from_bgr(bgr)
    except Exception as e:
        return None, str(e)
