# ============================================================
#   ОБЩИЙ ПАРАМЕТРИЗОВАННЫЙ ДВИЖОК ОБРАБОТКИ ЧЕКОВ
#   (патч: добавлен mode="auto" — самостоятельный подбор канала)
# ============================================================

import numpy as np
import cv2


def _get(d, path, default):
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
    """Гасит артефакт блочности JPEG (период ровно ksize x ksize по
    пикселям — проверено напрямую на исходных фото). Работает потому,
    что сумма любых ksize подряд идущих отсчётов ksize-периодичного
    сигнала — константа, независимо от фазы/сдвига решётки на фото."""
    ksize = max(2, int(ksize))
    return cv2.blur(gray.astype(np.float32), (ksize, ksize))


def _otsu_score(u8: np.ndarray) -> float:
    """Межклассовая дисперсия по Оцу — грубая, но дешёвая метрика
    'насколько хорошо сигнал делится на два кластера' (текст/фон).
    Чем выше — тем увереннее threshold нашёл настоящую границу, а не
    шум. Используется только для авто-выбора канала, не для самого
    порога."""
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = u8 >= thr
    if mask.all() or (~mask).all():
        return 0.0
    fg = u8[mask].astype(np.float32)
    bg = u8[~mask].astype(np.float32)
    w_fg = fg.size / u8.size
    w_bg = 1 - w_fg
    return float(w_fg * w_bg * (fg.mean() - bg.mean()) ** 2)


# Кандидаты сигнала для авто-режима. Каждый — это (mode, доп.параметры),
# в том же формате, что understand apply_recipe. Список специально
# небольшой: это самые частые "победители" среди 25 старых пресетов
# (см. LEGACY_PRESETS) плюс серый.
AUTO_CANDIDATES = [
    {"mode": "gray", "auto_invert": True},
    {"mode": "channel", "channel": {"r": -1, "g": 0, "b": 1}},   # B-R
    {"mode": "channel", "channel": {"r": 1, "g": -1, "b": 0}},   # R-G
    {"mode": "channel", "channel": {"r": -1, "g": 1, "b": 0}},   # G-R
    {"mode": "lab_b"},
]


def _base_signal(bgr: np.ndarray, params: dict) -> np.ndarray:
    """Шаг 1 из старого apply_recipe, вынесенный отдельно, чтобы им
    мог пользоваться и обычный режим, и auto."""
    mode = _get(params, 'mode', 'channel')
    if mode == 'channel':
        b, g, r = cv2.split(bgr.astype(np.float32))
        cr = _get(params, 'channel.r', -1)
        cg = _get(params, 'channel.g', 0)
        cb = _get(params, 'channel.b', 1)
        return cr * r + cg * g + cb * b
    elif mode == 'lab_b':
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        _, _, signal = cv2.split(lab)
        return signal
    else:  # 'gray'
        signal = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if _get(params, 'auto_invert', True):
            if float(np.mean(signal)) > 140:
                signal = 255 - signal
        return signal


def auto_pick_signal(bgr: np.ndarray, deblock_ksize: int = 8, lo: float = 1, hi: float = 99):
    """Перебирает AUTO_CANDIDATES, для каждого делает деблок+растяжку
    и оценивает контраст по Оцу. Возвращает (лучшая_картинка_uint8,
    имя_победившего_кандидата) — второе полезно для логов/статистики
    'какой канал чаще всего выигрывает', чтобы со временем можно было
    сократить список fon_type вручную."""
    best_img, best_score, best_name = None, -1.0, None
    for cand in AUTO_CANDIDATES:
        signal = _base_signal(bgr, cand)
        signal = _deblock8(signal, deblock_ksize)
        out = _pct_stretch(signal, lo, hi)
        score = _otsu_score(out)
        if score > best_score:
            best_img, best_score, best_name = out, score, cand.get("channel", cand["mode"])
    return best_img, best_name


def apply_recipe(bgr: np.ndarray, params: dict) -> np.ndarray:
    params = params or {}
    mode = _get(params, 'mode', 'channel')

    # --- шаг 0: raw — без обработки вообще ---
    if mode == 'raw':
        return bgr

    # --- новый режим: авто-подбор канала ---
    if mode == 'auto':
        ksize = _get(params, 'deblock.ksize', 8)
        lo = _get(params, 'percentile.lo', 1)
        hi = _get(params, 'percentile.hi', 99)
        out, _name = auto_pick_signal(bgr, deblock_ksize=ksize, lo=lo, hi=hi)
        return out

    # --- шаг 1: базовый сигнал (старое поведение) ---
    signal = _base_signal(bgr, params)

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

    # --- шаг 7: морфология (только если был порог) ---
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
