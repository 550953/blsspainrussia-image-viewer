import os
import io
import base64
import threading
from collections import OrderedDict

import requests
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import pandas as pd
import uvicorn
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#   СЕКРЕТЫ: источник истины — Infisical (Machine Identity).
#   Обычные переменные окружения Render используются ТОЛЬКО как
#   аварийный фоллбэк, если Infisical реально недоступен —
#   их можно оставить пустыми, если хочешь, чтобы секреты шли
#   строго из Infisical.
# ============================================================
INFISICAL_CLIENT_ID = os.environ.get("INFISICAL_CLIENT_ID")
INFISICAL_CLIENT_SECRET = os.environ.get("INFISICAL_CLIENT_SECRET")
INFISICAL_PROJECT_ID = os.environ.get("INFISICAL_PROJECT_ID")
INFISICAL_ENVIRONMENT = os.environ.get("ENVIRONMENT") or os.environ.get("INFISICAL_ENVIRONMENT", "dev")
INFISICAL_HOST = os.environ.get("INFISICAL_HOST", "https://app.infisical.com")
INFISICAL_SECRET_PATH = os.environ.get("INFISICAL_SECRET_PATH", "/")

SECRETS_SOURCE = "none"
# подробная диагностика по каждому ключу — видно в /api/status,
# что конкретно пошло не так (нет доступа, неверное имя, неверный
# environment_slug и т.д.)
SECRETS_DEBUG = {}


def _infisical_secret_name(key: str) -> str:
    """Имя секрета в Infisical можно переопределить переменной
    INFISICAL_SECRET_NAME_<KEY>, если у тебя там другое название
    (например, ты хранишь его как SUPABASE_URL_kj123664)."""
    return os.environ.get(f"INFISICAL_SECRET_NAME_{key}", key)


def _fetch_from_infisical(client, key: str):
    """Возвращает (value, error_message)."""
    secret_name = _infisical_secret_name(key)
    try:
        secret = client.secrets.get_secret_by_name(
            secret_name=secret_name,
            project_id=INFISICAL_PROJECT_ID,
            environment_slug=INFISICAL_ENVIRONMENT,
            secret_path=INFISICAL_SECRET_PATH,
        )
        value = (
            getattr(secret, "secretValue", None)
            or getattr(secret, "secret_value", None)
            or (secret.get("secretValue") if isinstance(secret, dict) else None)
        )
        if value:
            return value, None
        return None, f"секрет '{secret_name}' найден, но значение пустое"
    except Exception as e:
        # str(e) у infisical-исключений часто пустой — берём тип + repr,
        # плюс тело ответа, если это HTTP-ошибка (там обычно и есть причина:
        # 401 — неверный client_id/secret, 403 — нет доступа к проекту/окружению,
        # 404 — секрет с таким именем/path/environment не найден).
        detail = repr(e)
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                detail += f" | HTTP {resp.status_code}: {resp.text[:300]}"
            except Exception:
                pass
        return None, f"'{secret_name}' -> {type(e).__name__}: {detail}"


def load_secrets() -> dict:
    global SECRETS_SOURCE, SECRETS_DEBUG
    keys = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "APP_USER", "APP_PASS"]
    values = {}
    debug = {}

    if not (INFISICAL_CLIENT_ID and INFISICAL_CLIENT_SECRET and INFISICAL_PROJECT_ID):
        msg = "INFISICAL_CLIENT_ID / INFISICAL_CLIENT_SECRET / INFISICAL_PROJECT_ID не заданы в env"
        print(f"[infisical] {msg}")
        debug["_connection"] = msg
    else:
        try:
            from infisical_sdk import InfisicalSDKClient

            client = InfisicalSDKClient(host=INFISICAL_HOST)
            client.auth.universal_auth.login(INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET)
            print(
                f"[infisical] авторизация прошла успешно, "
                f"project_id={INFISICAL_PROJECT_ID}, environment_slug={INFISICAL_ENVIRONMENT}, "
                f"secret_path={INFISICAL_SECRET_PATH}"
            )

            got_any = False
            for key in keys:
                value, error = _fetch_from_infisical(client, key)
                if value:
                    values[key] = value
                    got_any = True
                    debug[key] = "ok"
                    print(f"[infisical] '{_infisical_secret_name(key)}' получен успешно")
                else:
                    debug[key] = error
                    print(f"[infisical] {error}")

            SECRETS_SOURCE = "infisical" if got_any else "infisical_failed"
        except Exception as e:
            msg = f"подключение/авторизация в Infisical не удались: {type(e).__name__}: {e!r}"
            print(f"[infisical] {msg}")
            debug["_connection"] = msg

    # Render env — только фоллбэк для того, чего не удалось достать из Infisical
    used_render_fallback = []
    for key in keys:
        if not values.get(key):
            env_value = os.environ.get(key)
            if env_value:
                values[key] = env_value
                used_render_fallback.append(key)
    if used_render_fallback:
        print(f"[secrets] взято из Render env как фоллбэк: {used_render_fallback}")
        SECRETS_SOURCE = "mixed (infisical + render_env)" if SECRETS_SOURCE == "infisical" else "render_env"

    SECRETS_DEBUG = debug
    return values


_SECRETS = load_secrets()
SUPABASE_URL = _SECRETS.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _SECRETS.get("SUPABASE_SERVICE_KEY")
APP_USER = _SECRETS.get("APP_USER") or "admin"
APP_PASS = _SECRETS.get("APP_PASS") or "changeme"


# ============================================================
#           ОБЩИЙ ПАРОЛЬ НА ВСЁ ПРИЛОЖЕНИЕ (HTTP Basic)
# ============================================================
class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("Authorization")
        if auth:
            try:
                scheme, credentials = auth.split(" ", 1)
                if scheme.lower() == "basic":
                    decoded = base64.b64decode(credentials).decode("utf-8")
                    username, _, password = decoded.partition(":")
                    if username == APP_USER and password == APP_PASS:
                        return await call_next(request)
            except Exception:
                pass
        return StarletteResponse(
            content="Требуется авторизация",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="price-viewer"'},
        )


app.add_middleware(BasicAuthMiddleware)


# ============================================================
#                        SUPABASE
# ============================================================
supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def require_supabase():
    if supabase is None:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY не заданы (ни в Infisical, ни в переменных Render)"
        )


def get_filename(url: str) -> str:
    if not url:
        return ""
    clean = str(url).split("?")[0]
    return clean.rsplit("/", 1)[-1]


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def api_status():
    """Небольшой диагностический эндпоинт — видно, откуда взялись секреты
    и реально ли приложение читает/пишет в Supabase."""
    status = {
        "secrets_source": SECRETS_SOURCE,
        "secrets_debug": SECRETS_DEBUG,
        "infisical_project_id": INFISICAL_PROJECT_ID,
        "infisical_environment": INFISICAL_ENVIRONMENT,
        "infisical_secret_path": INFISICAL_SECRET_PATH,
        "infisical_secret_names": {
            k: _infisical_secret_name(k)
            for k in ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "APP_USER", "APP_PASS"]
        },
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "supabase_connected": False,
        "row_count": None,
    }
    if supabase is not None:
        try:
            result = supabase.table("price_checks").select("id", count="exact").limit(1).execute()
            status["supabase_connected"] = True
            status["row_count"] = result.count
        except Exception as e:
            status["error"] = str(e)
    return status


# ============================================================
#          ПРОСТОЙ IN-MEMORY LRU-КЭШ (байты фото / готовые фильтры)
#   Снимает повторные скачивания с Backblaze и повторный прогон
#   через OpenCV при повторном открытии тех же фото.
# ============================================================
class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.data = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]

    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.maxsize:
                self.data.popitem(last=False)


image_bytes_cache = LRUCache(maxsize=250)   # сырые байты фото с Backblaze
filters_cache = LRUCache(maxsize=60)        # готовые варианты обработки (тяжелее по памяти)

IMAGE_CACHE_CONTROL = "public, max-age=604800, immutable"  # неделя — фото по URL не меняются
FILTERS_CACHE_CONTROL = "public, max-age=300"  # 5 минут — набор фильтров ещё дорабатывается,
                                                # без immutable, чтобы браузер сам перезапрашивал


# ============================================================
#                    ЗАГРУЗКА КАРТИНКИ
# ============================================================
def fetch_image_bytes(url: str) -> bytes:
    cached = image_bytes_cache.get(url)
    if cached is not None:
        return cached
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content
    image_bytes_cache.set(url, content)
    return content


@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        content = fetch_image_bytes(url)
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": IMAGE_CACHE_CONTROL},
        )
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": f"Фото не найдено: {e}"})


# ============================================================
#              ФИЛЬТРЫ ДЛЯ ПРОВЕРКИ ЧИТАЕМОСТИ
#
#   Набор пересобран после разбора конкретных "бледных" чеков
#   (низкий контраст, почти белый фон, ~150x80px, тяжёлый JPEG).
#   Раньше половина фильтров растягивала разницу каналов
#   ФИКСИРОВАННЫМИ alpha/beta — на таких бледных фото это либо
#   клипует всё в чистый чёрный/белый, либо не даёт контраста.
#   Сейчас база — адаптивная percentile-растяжка (_pct_stretch),
#   она сама подстраивается под реальный разброс яркости каждого
#   конкретного фото, а не бьёт по одному и тому же коэффициенту
#   для всех. Дублирующие "почти одинаковые" каналы (R-B и B-R и
#   т.п. отличаются только инверсией) убраны — их было 6-7 штук,
#   несущих одну и ту же информацию.
# ============================================================
def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _auto_invert(gray: np.ndarray) -> np.ndarray:
    return 255 - gray if float(np.mean(gray)) > 140 else gray


def _pct_stretch(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Растяжка гистограммы по процентилям — подстраивается под контраст
    конкретного фото вместо фиксированных alpha/beta."""
    lo_v, hi_v = np.percentile(arr, [lo, hi])
    out = (arr.astype(np.float32) - lo_v) / (hi_v - lo_v + 1e-6) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def _deblock8(gray: np.ndarray) -> np.ndarray:
    """Убирает JPEG-блочность период-8 (сильно сжатые почти-однотонные фото
    часто показывают решётку ровно 8x8 px — это не текстура бумаги и не шум,
    а артефакт квантования DCT-блоков). Обычный blur/denoise его не берёт,
    т.к. это не случайный шум, а жёсткая периодика — box(8,8) усредняет
    ровно по периоду и гасит её почти полностью."""
    return cv2.blur(gray.astype(np.float32), (8, 8))


def f_original(bgr):
    return bgr


def f_fast_clahe(bgr):
    base = _auto_invert(_to_gray(bgr))
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    return clahe.apply(base)


def f_adaptive_threshold(bgr):
    base = _auto_invert(_to_gray(bgr))
    binary = cv2.adaptiveThreshold(
        base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def f_br_stretch(bgr):
    """B-R разница с адаптивной percentile-растяжкой. Основной рабочий
    вариант для бледных/белесых чеков — самый согласованный сигнал
    в тестах на низкоконтрастных фото."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    return _pct_stretch(b - r)


def f_otsu_br(bgr):
    """Самый чистый бинарный результат в тестах — Otsu поверх
    percentile-растянутой B-R разницы."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    stretched = _pct_stretch(b - r)
    _, otsu = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def f_bilateral_otsu_br(bgr):
    """То же + bilateral перед порогом — гасит мелкий JPEG-блочный шум,
    сохраняя края букв."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    stretched = _pct_stretch(b - r)
    smooth = cv2.bilateralFilter(stretched, d=5, sigmaColor=30, sigmaSpace=30)
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def f_clahe_br(bgr):
    """CLAHE поверх уже растянутой B-R — усиливает локальный контраст
    без пересветов, которые давал CLAHE на сыром канале."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    stretched = _pct_stretch(b - r)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(stretched)


def f_lab_b_stretch(bgr):
    """L*a*b* b-канал (жёлто-синяя ось) — иногда ловит то, что
    RGB-разности замыливают, особенно на желтоватых/бежевых фото."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, _, b_channel = cv2.split(lab)
    return _pct_stretch(b_channel)


def f_upscale_sharpen(bgr):
    """Апскейл x3 поверх percentile-растянутой B-R + резкость —
    удобно для финального визуального разглядывания."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    stretched = _pct_stretch(b - r)
    h, w = stretched.shape
    big = cv2.resize(stretched, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(big, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(big, 1.8, blurred, -0.8, 0)
    return sharp


def f_deblock_stretch(bgr):
    """Деблокинг (box 8x8) + растяжка по сером — для фото с явной
    8-пиксельной решёткой JPEG-компрессии на фоне (не текстура бумаги)."""
    gray = _to_gray(bgr)
    deblocked = _deblock8(gray)
    return _pct_stretch(deblocked)


def f_deblock_clahe(bgr):
    """Деблокинг + CLAHE — локальный контраст без решётки поверх."""
    gray = _to_gray(bgr)
    deblocked = _deblock8(gray)
    stretched = _pct_stretch(deblocked)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    return clahe.apply(stretched)


def f_deblock_otsu(bgr):
    """Деблокинг + Otsu — бинарный вариант для тех же зернистых фото."""
    gray = _to_gray(bgr)
    deblocked = _deblock8(gray)
    stretched = _pct_stretch(deblocked)
    _, otsu = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


FILTERS = [
    ("original", "Оригинал", f_original),
    ("fast_clahe", "CLAHE (по серому)", f_fast_clahe),
    ("adaptive_threshold", "Адаптивная бинаризация", f_adaptive_threshold),
    ("br_stretch", "B-R растяжка (основной)", f_br_stretch),
    ("otsu_br", "B-R + Otsu (самый чистый)", f_otsu_br),
    ("bilateral_otsu_br", "B-R + шумодав + Otsu", f_bilateral_otsu_br),
    ("clahe_br", "B-R + CLAHE", f_clahe_br),
    ("lab_b_stretch", "LAB b-канал", f_lab_b_stretch),
    ("upscale_sharpen", "Апскейл x3 + резкость", f_upscale_sharpen),
    ("deblock_stretch", "Деблок 8x8 + растяжка", f_deblock_stretch),
    ("deblock_clahe", "Деблок 8x8 + CLAHE", f_deblock_clahe),
    ("deblock_otsu", "Деблок 8x8 + Otsu", f_deblock_otsu),
]


def encode_png_base64(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/proxy-filters")
async def proxy_filters(url: str):
    cached = filters_cache.get(url)
    if cached is not None:
        return Response(
            content=cached,
            media_type="application/json",
            headers={"Cache-Control": FILTERS_CACHE_CONTROL},
        )

    try:
        content = fetch_image_bytes(url)
        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": f"Фото не найдено: {e}"})

    results = []
    for key, label, func in FILTERS:
        try:
            out = func(bgr)
            results.append({"key": key, "label": label, "image": encode_png_base64(out)})
        except Exception as e:
            results.append({"key": key, "label": f"{label} — ошибка: {e}", "image": None})

    payload = {"width": bgr.shape[1], "height": bgr.shape[0], "filters": results}

    import json
    body = json.dumps(payload).encode("utf-8")
    filters_cache.set(url, body)

    return Response(content=body, media_type="application/json", headers={"Cache-Control": FILTERS_CACHE_CONTROL})


# ============================================================
#              ДАННЫЕ — всё живёт в Supabase
# ============================================================
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Импорт CSV — резервный/дополнительный способ занести новые файлы.
    Строки с уже существующим filename в базе НЕ трогаются (не затираем разметку)."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Ошибка чтения CSV: {str(e)}"})

    url_col = 'test' if 'test' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    existing = supabase.table("price_checks").select("filename").execute()
    existing_names = {r["filename"] for r in existing.data}

    rows_to_insert = []
    seen_in_file = set()
    for _, row in df.iterrows():
        url = str(row[url_col])
        fname = get_filename(url)
        if not fname or fname in existing_names or fname in seen_in_file:
            continue
        seen_in_file.add(fname)
        label = str(row[label_col])
        rows_to_insert.append({
            "filename": fname,
            "url": url,
            "label": label,
            "original_label": label,
            "comment": None,
            "is_broken": False,
        })

    inserted = 0
    for i in range(0, len(rows_to_insert), 500):
        chunk = rows_to_insert[i:i + 500]
        supabase.table("price_checks").insert(chunk).execute()
        inserted += len(chunk)

    skipped = len(df) - inserted
    return {"message": f"Добавлено новых строк: {inserted}. Пропущено (уже было в базе): {skipped}"}


@app.get("/api/data")
async def get_data():
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    all_rows = []
    page_size = 1000
    start = 0
    while True:
        result = (
            supabase.table("price_checks")
            .select("*")
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        chunk = result.data
        all_rows.extend(chunk)
        if len(chunk) < page_size:
            break
        start += page_size
    return all_rows


@app.post("/api/update")
async def update_row(payload: dict):
    """Точечное обновление одной строки: label / comment / is_broken."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    row_id = payload.get("id")
    if row_id is None:
        return JSONResponse(status_code=400, content={"error": "Нужен id"})

    update_fields = {}
    for key in ("label", "comment", "is_broken", "reviewed", "best_filter", "is_placeholder"):
        if key in payload:
            update_fields[key] = payload[key]

    if not update_fields:
        return JSONResponse(status_code=400, content={"error": "Нечего обновлять"})

    supabase.table("price_checks").update(update_fields).eq("id", row_id).execute()
    return {"message": "ok"}


@app.get("/download-csv")
async def download_csv():
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    result = supabase.table("price_checks").select("*").order("id").execute()
    if not result.data:
        return JSONResponse(status_code=404, content={"error": "Нет данных"})

    df = pd.DataFrame(result.data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)

    return StreamingResponse(
        buffer, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=price_checks_export.csv"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
