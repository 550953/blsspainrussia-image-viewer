import os
import io
import re
import gc
import time
import base64
import threading
from collections import OrderedDict
from typing import Optional

import requests
import numpy as np
import cv2

# Render free tier = 0.1 shared vCPU. Без этого numpy/OpenCV/Tesseract
# каждый норовят открыть свой пул потоков — на одном слабом ядре это
# только съедает время на переключение контекста и лишнюю память,
# выигрыша в скорости всё равно нет.
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # тот же лимит для tesseract (LSTM тоже через OpenMP)

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from pydantic import BaseModel
import pandas as pd
import uvicorn
from supabase import create_client, Client

import datetime
from filters_engine import apply_recipe
# Локальный pytesseract (ocr_engine.py) больше не используется в эндпоинтах —
# распознавание теперь идёт через внешний сервис (см. OCR_REMOTE_URL ниже).
# Импорт оставлен закомментированным на случай, если понадобится откат:
# from ocr_engine import extract_digits_from_array, extract_price_from_bgr

app = FastAPI()

# Время старта процесса — маячок для тестирования деплоя/OCR: если после
# git push это значение в /api/status не поменялось, значит крутится
# СТАРЫЙ процесс (деплой не подхватился), а не новый код с удалённым OCR.
PROCESS_STARTED_AT = datetime.datetime.utcnow().isoformat() + "Z"

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
    # Render (и любой другой оркестратор) стучится на health-check БЕЗ
    # заголовка Authorization — если этот путь тоже требует Basic Auth,
    # проверка всегда получает 401, Render считает деплой нездоровым
    # (именно это видно в логе: "GET /health HTTP/1.1 401 Unauthorized"
    # по кругу). Поэтому health-check — единственное публичное исключение.
    PUBLIC_PATHS = {"/health"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
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


@app.get("/health")
async def health():
    """Публичный health-check для Render (без Basic Auth, без обращений к
    Supabase/Infisical) — просто подтверждает, что процесс жив и отвечает."""
    return {"status": "ok"}


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


@app.get("/filters", response_class=HTMLResponse)
async def filters_page():
    with open("filters.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/broken", response_class=HTMLResponse)
async def broken_page():
    with open("broken.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def api_status():
    """Небольшой диагностический эндпоинт — видно, откуда взялись секреты
    и реально ли приложение читает/пишет в Supabase, и доступен ли Tesseract."""
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
        "ocr_source": "remote",
        "ocr_remote_url": OCR_REMOTE_URL,
        "ocr_available": bool(OCR_REMOTE_URL),
        "ocr_error": None if OCR_REMOTE_URL else "OCR_REMOTE_URL не задан",
        # маячок: если это время не совпадает с реальным временем последнего
        # деплоя (Render → Events), значит смотрите на старый процесс, а не
        # на кэш браузера — POST-эндпоинты (/api/ocr-check и т.п.) браузер
        # не кеширует вовсе, а этот GET теперь явно помечен no-store ниже.
        "process_started_at": PROCESS_STARTED_AT,
    }
    if supabase is not None:
        try:
            result = supabase.table("price_checks").select("id", count="exact").limit(1).execute()
            status["supabase_connected"] = True
            status["row_count"] = result.count
        except Exception as e:
            status["error"] = str(e)

    return JSONResponse(content=status, headers={"Cache-Control": "no-store"})


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

IMAGE_CACHE_CONTROL = "public, max-age=604800, immutable"  # неделя — фото по URL не меняются


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


def fetch_image_bgr(url: str):
    """Скачивает (с кэшем) и декодирует в OpenCV BGR-массив. Возвращает
    None, если фото не скачалось/не декодировалось."""
    content = fetch_image_bytes(url)
    arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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


def encode_png_bytes(arr: np.ndarray) -> bytes:
    """PNG-байты (не base64) — то, что реально уходит в файл/по сети."""
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_png_base64(arr: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png_bytes(arr)).decode("ascii")


# ============================================================
#                 УДАЛЁННЫЙ OCR (bls.shikinn.com)
#
#   Вместо локального pytesseract (ему нужен системный tesseract-ocr,
#   ставится только через Docker-деплой) — шлём сырые байты фото на
#   ваш собственный OCR-сервис по HTTP.
#
#   ВАЖНО: точный контракт запроса/ответа bls.shikinn.com/ocr мне не
#   известен — ниже сделан наиболее типичный вариант:
#     запрос:  POST multipart/form-data, файл в поле OCR_REMOTE_FIELD_NAME
#     ответ:   JSON, распознанный текст в одном из полей
#              text / digits / result / price / value
#   Если сервис ждёт другое (например JSON с base64 вместо файла, или
#   другое имя поля/ответа) — правьте только эту функцию, остальной
#   код её не касается.
# ============================================================
OCR_REMOTE_URL = os.environ.get("OCR_REMOTE_URL", "https://bls.shikinn.com/ocr")
# Free-tier Render засыпает через ~15 мин простоя и просыпается 30-50 сек
# (см. комментарий в Dockerfile). 20 сек таймаута на это не хватает — запрос
# обрывался по таймауту, а следующий клик уже попадал в проснувшийся
# инстанс и работал, создавая ложное впечатление "откуда-то берёт ответ
# помимо этого сервиса". Поднял таймаут и добавил один автоповтор с ещё
# большим таймаутом именно на случай холодного старта.
OCR_REMOTE_TIMEOUT = float(os.environ.get("OCR_REMOTE_TIMEOUT", 45))
OCR_REMOTE_COLD_START_TIMEOUT = float(os.environ.get("OCR_REMOTE_COLD_START_TIMEOUT", 70))
OCR_REMOTE_FIELD_NAME = os.environ.get("OCR_REMOTE_FIELD_NAME", "file")


def _post_to_remote_ocr(image_bytes: bytes, timeout: float):
    return requests.post(
        OCR_REMOTE_URL,
        files={OCR_REMOTE_FIELD_NAME: ("image.jpg", image_bytes, "application/octet-stream")},
        timeout=timeout,
    )


def extract_via_remote_ocr(image_bytes: bytes):
    """Отправляет уже готовые байты картинки (PNG/JPEG — не важно) на
    внешний OCR и вытаскивает из ответа только цифры.
    Возвращает (digits_str_or_None, error_or_None)."""
    if not OCR_REMOTE_URL:
        return None, "OCR_REMOTE_URL не задан"

    t0 = time.monotonic()
    try:
        resp = _post_to_remote_ocr(image_bytes, OCR_REMOTE_TIMEOUT)
        resp.raise_for_status()
        print(f"[remote_ocr] ok за {time.monotonic() - t0:.1f}с, status={resp.status_code}")
    except requests.exceptions.Timeout:
        print(f"[remote_ocr] ТАЙМАУТ на {OCR_REMOTE_TIMEOUT:.0f}с (прошло {time.monotonic() - t0:.1f}с) — повтор с {OCR_REMOTE_COLD_START_TIMEOUT:.0f}с")
        # Похоже на холодный старт free-tier инстанса — даём ему ещё один
        # шанс с большим таймаутом, вместо того чтобы сразу сдаваться.
        t1 = time.monotonic()
        try:
            resp = _post_to_remote_ocr(image_bytes, OCR_REMOTE_COLD_START_TIMEOUT)
            resp.raise_for_status()
            print(f"[remote_ocr] повтор ok за {time.monotonic() - t1:.1f}с, status={resp.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[remote_ocr] повтор ПРОВАЛИЛСЯ за {time.monotonic() - t1:.1f}с: {type(e).__name__}: {e}")
            return None, (
                f"OCR-сервис не ответил даже за {OCR_REMOTE_COLD_START_TIMEOUT:.0f} сек "
                f"(похоже на холодный старт Render, но что-то пошло не так): {e}"
            )
    except requests.exceptions.RequestException as e:
        print(f"[remote_ocr] ОШИБКА за {time.monotonic() - t0:.1f}с: {type(e).__name__}: {e}")
        return None, f"OCR-сервис недоступен ({OCR_REMOTE_URL}): {e}"

    raw_text = None
    try:
        data = resp.json()
        if isinstance(data, dict):
            for key in ("text", "digits", "result", "price", "value"):
                if data.get(key):
                    raw_text = str(data[key])
                    break
            if raw_text is None:
                raw_text = str(data)
        else:
            raw_text = str(data)
    except ValueError:
        # ответ не JSON — берём как есть (вдруг это просто "1234" текстом)
        raw_text = resp.text

    digits = re.sub(r"\D", "", raw_text or "")
    if not digits:
        print(f"[remote_ocr] цифры не найдены, сырой ответ: {raw_text!r}")
        return None, f"цифры не найдены в ответе OCR-сервиса (сырой ответ: {raw_text[:200]!r})"
    print(f"[remote_ocr] распознано: {digits}")
    return digits, None


# ============================================================
#     ФИЛЬТРЫ КАК ДАННЫЕ: генератор превью по params + CRUD рецептов
#
#   Раньше здесь было 12 захардкоженных python-функций и статичная
#   сетка из 12 картинок. Теперь — один параметризованный движок
#   (filters_engine.apply_recipe), а сами варианты обработки живут
#   в таблице filter_presets и редактируются через страницу /filters
#   или ползунками прямо в карточке чека — без правки кода.
# ============================================================
class ApplyRecipeRequest(BaseModel):
    url: str
    params: dict


@app.post("/api/apply-recipe")
async def api_apply_recipe(payload: ApplyRecipeRequest):
    """Обрабатывает фото ОДНИМ набором параметров и возвращает PNG-превью.
    Дергается с фронта при каждом движении ползунка (с debounce ~200ms
    на стороне клиента, чтобы не заваливать сервер на каждый пиксель
    перетаскивания)."""
    try:
        bgr = fetch_image_bgr(payload.url)
        if bgr is None:
            return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})
        out = apply_recipe(bgr, payload.params)
        b64 = encode_png_base64(out)
        return {"image": b64, "width": bgr.shape[1], "height": bgr.shape[0]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Ошибка обработки: {e}"})


class RecipeIn(BaseModel):
    name: str
    description: Optional[str] = None
    params: dict
    fon_type: Optional[str] = None
    is_default: Optional[bool] = False
    sort_order: Optional[int] = 0


@app.get("/api/recipes")
async def list_recipes(fon_type: Optional[str] = None):
    """Список рецептов. Если передан fon_type — сначала рецепты под этот
    фон (свои + универсальные default), затем остальные — чтобы в модалке
    сверху показывались наиболее релевантные варианты."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    result = supabase.table("filter_presets").select("*").order("sort_order").execute()
    rows = result.data

    if fon_type:
        matching = [r for r in rows if r.get("fon_type") == fon_type]
        defaults = [r for r in rows if r.get("fon_type") is None and r.get("is_default")]
        rest = [r for r in rows if r not in matching and r not in defaults]
        rows = matching + defaults + rest

    return rows


@app.post("/api/recipes")
async def create_recipe(recipe: RecipeIn):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    result = supabase.table("filter_presets").insert(recipe.dict()).execute()
    return result.data[0]


@app.put("/api/recipes/{recipe_id}")
async def update_recipe(recipe_id: int, recipe: RecipeIn):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    payload = recipe.dict()
    payload["updated_at"] = "now()"
    supabase.table("filter_presets").update(payload).eq("id", recipe_id).execute()
    return {"message": "ok"}


@app.delete("/api/recipes/{recipe_id}")
async def delete_recipe(recipe_id: int):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    supabase.table("filter_presets").delete().eq("id", recipe_id).execute()
    return {"message": "ok"}


@app.post("/api/recipes/{recipe_id}/mark-used")
async def mark_recipe_used(recipe_id: int, payload: dict):
    """Привязывает рецепт к конкретному чеку (best_recipe_id) — для
    статистики 'что чаще всего спасает такой-то фон'."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    row_id = payload.get("id")
    if row_id is None:
        return JSONResponse(status_code=400, content={"error": "Нужен id чека"})
    supabase.table("price_checks").update({
        "best_recipe_id": recipe_id,
        "reviewed": True,
    }).eq("id", row_id).execute()
    return {"message": "ok"}


# ============================================================
#              СТРАНИЦА "ПРОБЛЕМНЫЕ" (/broken)
# ============================================================
@app.get("/api/broken")
async def api_broken():
    """Только реально отмеченные операторами забагованные чеки,
    сгруппированные по категории фона — для отдельной страницы,
    а не общей ленты из 5000+ карточек."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    all_rows = []
    start = 0
    page_size = 1000
    while True:
        result = (supabase.table("price_checks")
                  .select("*")
                  .eq("is_broken", True)
                  .order("fon_type")
                  .range(start, start + page_size - 1)
                  .execute())
        chunk = result.data
        all_rows.extend(chunk)
        if len(chunk) < page_size:
            break
        start += page_size

    groups = {}
    for row in all_rows:
        key = row.get("fon_type") or "(без категории)"
        groups.setdefault(key, []).append(row)

    # сортируем группы по размеру убыв. — сначала самые массовые проблемные фоны
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    return {
        "total": len(all_rows),
        "groups": [{"fon_type": k, "count": len(v), "rows": v} for k, v in ordered],
    }


# ============================================================
#                            OCR
#
#   Автоматическая проверка читаемости — вместо того, чтобы искать
#   новые плохие фото глазами, можно прогнать выборку через Tesseract
#   и получить список того, где цифры не читаются или не совпадают
#   с текущим label. Работает как поверх сырого фото (тем же
#   алгоритмом, что был в присланном ocr_engine.py), так и поверх
#   результата любого рецепта из filters_engine — чтобы проверить,
#   действительно ли конкретная обработка помогает распознаванию,
#   а не просто "на глаз читается".
#
#   ВАЖНО: pytesseract — это только Python-обёртка, ей нужен системный
#   бинарник tesseract-ocr. На Render 'Native' environment apt-get
#   недоступен (см. /api/status -> ocr_available) — для этой функции
#   нужен Docker-деплой, см. приложенный Dockerfile.
# ============================================================
class OcrCheckRequest(BaseModel):
    url: str
    params: Optional[dict] = None  # если не передано — используется свой встроенный clean_image


@app.post("/api/ocr-check")
async def api_ocr_check(payload: OcrCheckRequest):
    """Прогоняет OCR либо по сырому фото (свой алгоритм очистки), либо
    по результату конкретного рецепта — используется кнопкой
    '🤖 Проверить OCR' прямо в лаборатории фильтров."""
    try:
        bgr = fetch_image_bgr(payload.url)
        if bgr is None:
            return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})

        if payload.params:
            processed = apply_recipe(bgr, payload.params)
            digits, err = extract_via_remote_ocr(encode_png_bytes(processed))
        else:
            digits, err = extract_via_remote_ocr(fetch_image_bytes(payload.url))

        return JSONResponse(
            content={"digits": digits, "error": err},
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Ошибка OCR: {e}"}, headers={"Cache-Control": "no-store"})


class OcrScanRequest(BaseModel):
    limit: Optional[int] = 20
    fon_type: Optional[str] = None
    only_unreviewed: Optional[bool] = True
    recipe_id: Optional[int] = None  # если не задан — берём собственную очистку ocr_engine


# На фри тарифе Render 0.1 vCPU + 512 МБ RAM: один запрос "на 1000 фото
# разом" либо не влезет в память, либо просто не успеет ответить до того,
# как прокси/браузер решат, что сервис "упал" (таймаут). Поэтому:
#   - дефолт и потолок лимита сильно ниже, чем было (100, а не 1000);
#   - есть мягкий бюджет по времени — если скан идёт дольше OCR_SCAN_TIME_BUDGET_S,
#     останавливаемся и отдаём то, что успели, с пометкой stopped_early=True,
#     вместо того чтобы зависнуть до убийства процесса/запроса.
OCR_SCAN_MAX_LIMIT = int(os.environ.get("OCR_SCAN_MAX_LIMIT", 100))
OCR_SCAN_TIME_BUDGET_S = float(os.environ.get("OCR_SCAN_TIME_BUDGET_S", 25))


@app.post("/api/ocr-scan")
async def api_ocr_scan(payload: OcrScanRequest):
    """Пакетная проверка выборки чеков через OCR — это и есть 'тестовая
    выборка, которая сама покажет новые плохие фото', без ручного
    пролистывания. Результат сохраняется в price_checks (ocr_text,
    ocr_mismatch, ocr_checked_at), поэтому потом их можно отфильтровать
    прямо на главной странице ('🤖 OCR не совпадает' / '🤖 OCR не прочитал')."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    recipe_params = None
    if payload.recipe_id is not None:
        rr = supabase.table("filter_presets").select("params").eq("id", payload.recipe_id).limit(1).execute()
        if not rr.data:
            return JSONResponse(status_code=400, content={"error": f"Рецепт id={payload.recipe_id} не найден"})
        recipe_params = rr.data[0]["params"]

    query = supabase.table("price_checks").select("id, url, label, filename")
    if payload.fon_type:
        query = query.eq("fon_type", payload.fon_type)
    if payload.only_unreviewed:
        query = query.eq("reviewed", False)
    limit = max(1, min(int(payload.limit or 20), OCR_SCAN_MAX_LIMIT))
    result = query.order("id").limit(limit).execute()
    rows = result.data

    checked = 0
    mismatches = []
    unread = []
    started_at = time.monotonic()
    stopped_early = False
    for i, row in enumerate(rows):
        if time.monotonic() - started_at > OCR_SCAN_TIME_BUDGET_S:
            stopped_early = True
            break

        url = row.get("url")
        if not url:
            continue
        bgr = None
        processed = None
        try:
            bgr = fetch_image_bgr(url)
            if bgr is None:
                digits, err = None, "не удалось декодировать фото"
            elif recipe_params is not None:
                processed = apply_recipe(bgr, recipe_params)
                digits, err = extract_via_remote_ocr(encode_png_bytes(processed))
            else:
                digits, err = extract_via_remote_ocr(fetch_image_bytes(url))
        except Exception as e:
            digits, err = None, str(e)
        finally:
            # Явно освобождаем декодированные картинки — на 512 МБ RAM
            # не хотим ждать, пока сборщик мусора сам доберётся до них.
            del bgr, processed
            if i % 10 == 0:
                gc.collect()

        label = str(row.get("label") or "")
        is_mismatch = (digits is None) or (digits.lstrip("0") != label.lstrip("0"))

        supabase.table("price_checks").update({
            "ocr_text": digits,
            "ocr_mismatch": is_mismatch,
            "ocr_checked_at": "now()",
        }).eq("id", row["id"]).execute()

        checked += 1
        entry = {"id": row["id"], "filename": row.get("filename"), "label": label, "ocr_text": digits, "error": err}
        if digits is None:
            unread.append(entry)
        elif is_mismatch:
            mismatches.append(entry)

    return {
        "checked": checked,
        "requested": len(rows),
        "limit": limit,  # сколько реально запросили из БД в этом вызове (после clamp на OCR_SCAN_MAX_LIMIT)
        "stopped_early": stopped_early,  # True = уперлись в тайм-бюджет, а не кончились строки — жмите ещё раз
        "mismatches_count": len(mismatches),
        "unread_count": len(unread),
        "mismatches": mismatches[:50],
        "unread": unread[:50],
    }


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
    """Точечное обновление одной строки: label / comment / is_broken / и т.д."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    row_id = payload.get("id")
    if row_id is None:
        return JSONResponse(status_code=400, content={"error": "Нужен id"})

    update_fields = {}
    for key in ("label", "comment", "is_broken", "reviewed", "best_filter", "is_placeholder",
                "fon_type", "best_recipe_id"):
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
