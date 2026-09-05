# ============================================================
#   ОБЩИЙ ПАРАМЕТРИЗОВАННЫЙ ДВИЖОК ОБРАБОТКИ ЧЕКОВ
#
#   Раньше: 12 отдельных def f_xxx(bgr) функций — новый вариант
#   обработки = новый код = поход ко мне.
#
#   Теперь: ОДНА функция apply_recipe(bgr, params), где params —
#   это jsonb-запись из таблицы filter_presets. Новый вариант =
#   новая строка в базе, которую оператор сам создаёт через
#   страницу "Фильтры", двигая ползунки.
#
#   Все 12 старых функций из старого main.py — это ЧАСТНЫЕ СЛУЧАИ
#   данного пайплайна с конкретными params (см. LEGACY_PRESETS ниже,
#   используется как seed-данные при миграции).
#
#   Порядок пайплайна (шаги пропускаются, если выключены в params):
#     1. base signal:  channel-формула ИЛИ градации серого (+ auto-invert)
#                       ИЛИ LAB b-канал ИЛИ raw (без обработки)
#     2. деблокинг 8x8 (убирает JPEG-решётку на бледных фото)
#     3. percentile stretch -> uint8
#     4. CLAHE (локальный контраст)
#     5. bilateral (шумодав с сохранением краёв)
#     6. threshold: none / otsu / adaptive / manual
#     7. морфология (open+close, чистит бинаризацию от точек)
#     8. ручная инверсия (после всего, если нужно перевернуть ещё раз)
#     9. апскейл + резкость (финальный шаг для разглядывания глазами)
# ============================================================

import numpy as np
import cv2


def _get(d, path, default):
    """Достаёт вложенное значение из dict по точечному пути, либо default."""
    cur = d or {}
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def _pct_stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo = max(0, min(49, lo))
    hi = max(lo + 1, min(100, hi))
    lo_v, hi_v = np.percentile(arr, [lo, hi])
    out = (arr.astype(np.float32) - lo_v) / (hi_v - lo_v + 1e-6) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def _deblock8(gray: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(2, int(ksize))
    return cv2.blur(gray.astype(np.float32), (ksize, ksize))


def apply_recipe(bgr: np.ndarray, params: dict) -> np.ndarray:
    params = params or {}
    mode = _get(params, 'mode', 'channel')

    # --- шаг 0: raw — без обработки вообще ---
    if mode == 'raw':
        return bgr

    # --- шаг 1: базовый сигнал ---
    if mode == 'channel':
        b, g, r = cv2.split(bgr.astype(np.float32))
        cr = _get(params, 'channel.r', -1)
        cg = _get(params, 'channel.g', 0)
        cb = _get(params, 'channel.b', 1)
        signal = cr * r + cg * g + cb * b
    elif mode == 'lab_b':
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        _, _, signal = cv2.split(lab)
    else:  # 'gray'
        signal = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if _get(params, 'auto_invert', True):
            if float(np.mean(signal)) > 140:
                signal = 255 - signal

    # --- шаг 2: деблокинг (на float-сигнале, до нормализации) ---
    if _get(params, 'deblock.enabled', False):
        signal = _deblock8(signal, _get(params, 'deblock.ksize', 8))

    # --- шаг 3: нормализация в uint8 ---
    lo = _get(params, 'percentile.lo', 1)
    hi = _get(params, 'percentile.hi', 99)
    out = _pct_stretch(signal, lo, hi)

    # --- шаг 4: CLAHE ---
    if _get(params, 'clahe.enabled', False):
        clip = float(_get(params, 'clahe.clip', 3.5))
        tile = int(_get(params, 'clahe.tile', 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        out = clahe.apply(out)

    # --- шаг 5: bilateral ---
    if _get(params, 'bilateral.enabled', False):
        d = int(_get(params, 'bilateral.d', 5))
        sc = float(_get(params, 'bilateral.sigma_color', 30))
        ss = float(_get(params, 'bilateral.sigma_space', 30))
        out = cv2.bilateralFilter(out, d=d, sigmaColor=sc, sigmaSpace=ss)

    # --- шаг 6: порог ---
    thr_mode = _get(params, 'threshold.mode', 'none')
    if thr_mode == 'otsu':
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thr_mode == 'manual':
        val = int(_get(params, 'threshold.value', 128))
        _, out = cv2.threshold(out, val, 255, cv2.THRESH_BINARY)
    elif thr_mode == 'adaptive':
        block = int(_get(params, 'threshold.block_size', 15))
        if block % 2 == 0:
            block += 1
        c = float(_get(params, 'threshold.c', 3))
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, c)

    # --- шаг 7: морфология (только если был порог — на полутонах бессмысленна) ---
    if thr_mode != 'none' and _get(params, 'morphology.enabled', False):
        k = int(_get(params, 'morphology.ksize', 2))
        kernel = np.ones((k, k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    # --- шаг 8: ручная инверсия поверх всего ---
    if _get(params, 'manual_invert', False):
        out = 255 - out

    # --- шаг 9: апскейл + резкость ---
    factor = float(_get(params, 'upscale.factor', 1))
    if factor > 1.01:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
        sharpen = float(_get(params, 'upscale.sharpen', 0))
        if sharpen > 0:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
            out = cv2.addWeighted(out, 1 + sharpen, blurred, -sharpen, 0)

    return out


# ============================================================
#   SEED: старые 12 python-функций как стартовые рецепты.
#   Используется один раз скриптом 03_seed_legacy_presets.py —
#   дальше это обычные строки в filter_presets, редактируемые
#   через страницу "Фильтры" как любые новые.
# ============================================================
LEGACY_PRESETS = [
    {
        "name": "Оригинал",
        "description": "Без обработки — как есть.",
        "is_legacy": True, "is_default": True, "sort_order": 0,
        "params": {"mode": "raw"},
    },
    {
        "name": "CLAHE (по серому)",
        "description": "Базовый локальный контраст по серому с авто-инверсией.",
        "is_legacy": True, "sort_order": 10,
        "params": {"mode": "gray", "auto_invert": True,
                   "clahe": {"enabled": True, "clip": 3.5, "tile": 8}},
    },
    {
        "name": "Адаптивная бинаризация",
        "description": "Локальный порог + чистка морфологией.",
        "is_legacy": True, "sort_order": 20,
        "params": {"mode": "gray", "auto_invert": True,
                   "threshold": {"mode": "adaptive", "block_size": 15, "c": 3},
                   "morphology": {"enabled": True, "ksize": 2}},
    },
    {
        "name": "B-R растяжка (основной)",
        "description": "Разница синего и красного канала — рабочая лошадка для бледных чеков.",
        "is_legacy": True, "is_default": True, "sort_order": 30,
        "params": {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1},
                   "percentile": {"lo": 1, "hi": 99}},
    },
    {
        "name": "B-R + Otsu (самый чистый)",
        "description": "То же + автопорог Otsu — самый чистый бинарный результат в тестах.",
        "is_legacy": True, "is_default": True, "sort_order": 40,
        "params": {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1},
                   "percentile": {"lo": 1, "hi": 99},
                   "threshold": {"mode": "otsu"}},
    },
    {
        "name": "B-R + шумодав + Otsu",
        "description": "Bilateral перед порогом — гасит JPEG-шум, сохраняя края букв.",
        "is_legacy": True, "sort_order": 50,
        "params": {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1},
                   "percentile": {"lo": 1, "hi": 99},
                   "bilateral": {"enabled": True, "d": 5, "sigma_color": 30, "sigma_space": 30},
                   "threshold": {"mode": "otsu"}},
    },
    {
        "name": "B-R + CLAHE",
        "description": "Локальный контраст поверх растянутой B-R без пересветов.",
        "is_legacy": True, "sort_order": 60,
        "params": {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1},
                   "percentile": {"lo": 1, "hi": 99},
                   "clahe": {"enabled": True, "clip": 4.0, "tile": 4}},
    },
    {
        "name": "LAB b-канал",
        "description": "Жёлто-синяя ось LAB — ловит то, что RGB-разности замыливают.",
        "is_legacy": True, "sort_order": 70,
        "params": {"mode": "lab_b", "percentile": {"lo": 1, "hi": 99}},
    },
    {
        "name": "Апскейл x3 + резкость",
        "description": "Для финального визуального разглядывания.",
        "is_legacy": True, "sort_order": 80,
        "params": {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1},
                   "percentile": {"lo": 1, "hi": 99},
                   "upscale": {"factor": 3, "sharpen": 0.8}},
    },
    {
        "name": "Деблок 8x8 + растяжка",
        "description": "Убирает JPEG-решётку на бледном фоне.",
        "is_legacy": True, "sort_order": 90,
        "params": {"mode": "gray", "auto_invert": False,
                   "deblock": {"enabled": True, "ksize": 8},
                   "percentile": {"lo": 1, "hi": 99}},
    },
    {
        "name": "Деблок 8x8 + CLAHE",
        "description": "Деблокинг + локальный контраст без решётки поверх.",
        "is_legacy": True, "sort_order": 100,
        "params": {"mode": "gray", "auto_invert": False,
                   "deblock": {"enabled": True, "ksize": 8},
                   "percentile": {"lo": 1, "hi": 99},
                   "clahe": {"enabled": True, "clip": 4.0, "tile": 4}},
    },
    {
        "name": "Деблок 8x8 + Otsu",
        "description": "Бинарный вариант для тех же зернистых фото.",
        "is_legacy": True, "sort_order": 110,
        "params": {"mode": "gray", "auto_invert": False,
                   "deblock": {"enabled": True, "ksize": 8},
                   "percentile": {"lo": 1, "hi": 99},
                   "threshold": {"mode": "otsu"}},
    },
]
