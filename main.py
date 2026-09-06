import asyncio
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


def _post_to_remote_ocr(image_bytes: bytes, timeout: float):
    """ВАЖНО: bls.shikinn.com/ocr ждёт JSON {"images": "<base64>"},
    НЕ multipart/form-data-файл — так у него написан OCRRequest.
    Раньше здесь слался файл, из-за чего сервис падал с 500 (см. его
    собственный traceback: RequestValidationError на теле запроса)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return requests.post(
        OCR_REMOTE_URL,
        json={"images": b64},
        timeout=timeout,
    )


def _retry_after_seconds(resp, default: float = 5.0, cap: float = 15.0) -> float:
    """Читает Retry-After из ответа 429, если сервер его прислал."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    if header:
        try:
            return min(float(header), cap)
        except ValueError:
            pass
    return default


def extract_via_remote_ocr(image_bytes: bytes):
    """Отправляет уже готовые байты картинки на внешний OCR и вытаскивает
    из ответа только цифры. Возвращает (digits_str_or_None, error_or_None).

    Контракт bls.shikinn.com/ocr (из его собственного main.py):
      запрос:  POST JSON {"images": "<base64>"} (или список base64)
      ответ:   {"results": [{"text": "353", "source": "..."}]}
    "text" == "0" у этого сервиса означает "не распознал" (он и сам так
    его использует как sentinel), поэтому трактуем "0" как отсутствие
    результата, а не как настоящий распознанный ноль.

    ОГРАНИЧЕНИЕ этого конкретного сервиса: он принудительно приводит
    результат ровно к 3 цифрам (заточен под 3-значные капчи), всё
    остальное превращается в "0". Если у вас цены не всегда из 3 цифр,
    этот сервис для них всегда будет отвечать "не распознал"."""
    if not OCR_REMOTE_URL:
        return None, "OCR_REMOTE_URL не задан"

    t0 = time.monotonic()
    try:
        resp = _post_to_remote_ocr(image_bytes, OCR_REMOTE_TIMEOUT)
        if resp.status_code == 429:
            wait_s = _retry_after_seconds(resp)
            print(f"[remote_ocr] 429 Too Many Requests — жду {wait_s:.1f}с и пробую ещё раз")
            time.sleep(wait_s)
            t0 = time.monotonic()
            resp = _post_to_remote_ocr(image_bytes, OCR_REMOTE_TIMEOUT)
            if resp.status_code == 429:
                print(f"[remote_ocr] снова 429 после ожидания {wait_s:.1f}с — сдаюсь")
                return None, (
                    "OCR-сервис ограничивает частоту запросов (429 Too Many Requests) — "
                    "похоже, у bls.shikinn.com/ocr включён rate limit. Подождите между "
                    "проверками или ослабьте лимит на самом сервисе."
                )
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

    try:
        data = resp.json()
        results = data.get("results") or []
        raw_text = results[0].get("text") if results else None
        source = results[0].get("source") if results else None
    except (ValueError, IndexError, AttributeError) as e:
        print(f"[remote_ocr] не удалось разобрать ответ: {e}, сырой ответ: {resp.text[:200]!r}")
        return None, f"неожиданный формат ответа OCR-сервиса: {e}"

    if not raw_text or raw_text == "0":
        print(f"[remote_ocr] сервис не распознал (source={source})")
        return None, (
            "OCR-сервис не смог распознать (или число не из 3 цифр — "
            f"этот сервис принудительно ждёт ровно 3 цифры; source={source})"
        )

    digits = re.sub(r"\D", "", raw_text)
    if not digits:
        print(f"[remote_ocr] в ответе нет цифр: {raw_text!r}")
        return None, f"в ответе OCR нет цифр: {raw_text!r}"
    print(f"[remote_ocr] распознано: {digits} (source={source})")
    return digits, None


def extract_via_remote_ocr_batch(images_bytes: list):
    """То же самое, что extract_via_remote_ocr, но ОДНИМ HTTP-запросом
    на N картинок сразу — именно под это спроектирован bls.shikinn.com/ocr
    (contact-sheet, до 500 картинок, ~12-25 шт/сек по замерам). Раньше
    фронт бил его по одной картинке за раз (checkCell -> /api/ocr-check
    в цикле по всем фильтрам строки), из-за чего каждая ячейка платила
    полный цикл ключ->прокси->Gemini ради одной картинки, и вся строка
    последовательно копила эти 1.5-3.5с (а при холодном старте/rate-limit
    — 45-70с таймауты) вместо одного быстрого батча.

    Возвращает список (digits_or_None, source_or_None, error_or_None) —
    в том же порядке, что images_bytes."""
    if not images_bytes:
        return []
    if not OCR_REMOTE_URL:
        return [(None, None, "OCR_REMOTE_URL не задан")] * len(images_bytes)

    b64_list = [base64.b64encode(b).decode("ascii") for b in images_bytes]
    t0 = time.monotonic()
    try:
        resp = requests.post(OCR_REMOTE_URL, json={"images": b64_list}, timeout=OCR_REMOTE_TIMEOUT)
        if resp.status_code == 429:
            wait_s = _retry_after_seconds(resp)
            print(f"[remote_ocr_batch] 429 — жду {wait_s:.1f}с и пробую ещё раз")
            time.sleep(wait_s)
            resp = requests.post(OCR_REMOTE_URL, json={"images": b64_list}, timeout=OCR_REMOTE_TIMEOUT)
        resp.raise_for_status()
        print(f"[remote_ocr_batch] ok за {time.monotonic() - t0:.1f}с, {len(images_bytes)} шт")
    except requests.exceptions.Timeout:
        print(f"[remote_ocr_batch] ТАЙМАУТ на {OCR_REMOTE_TIMEOUT:.0f}с — повтор с {OCR_REMOTE_COLD_START_TIMEOUT:.0f}с (холодный старт?)")
        try:
            resp = requests.post(OCR_REMOTE_URL, json={"images": b64_list}, timeout=OCR_REMOTE_COLD_START_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return [(None, None, f"OCR-сервис не ответил даже за {OCR_REMOTE_COLD_START_TIMEOUT:.0f}с: {e}")] * len(images_bytes)
    except requests.exceptions.RequestException as e:
        print(f"[remote_ocr_batch] ОШИБКА за {time.monotonic() - t0:.1f}с: {type(e).__name__}: {e}")
        return [(None, None, f"OCR-сервис недоступен ({OCR_REMOTE_URL}): {e}")] * len(images_bytes)

    try:
        data = resp.json()
        results = data.get("results") or []
    except ValueError as e:
        print(f"[remote_ocr_batch] не удалось разобрать ответ: {e}, сырой ответ: {resp.text[:200]!r}")
        return [(None, None, f"неожиданный формат ответа OCR-сервиса: {e}")] * len(images_bytes)

    out = []
    for i in range(len(images_bytes)):
        item = results[i] if i < len(results) else {}
        raw_text = (item or {}).get("text")
        source = (item or {}).get("source")
        if not raw_text or raw_text == "0":
            out.append((None, source, f"не распознано (source={source})"))
            continue
        digits = re.sub(r"\D", "", raw_text)
        out.append((digits, source, None) if digits else (None, source, f"в ответе нет цифр: {raw_text!r}"))
    return out


# ============================================================
#              БЫСТРЫЙ СТАТУС OCR-ДОНОРА (bls.shikinn.com)
#
#   Раньше единственный способ узнать "спит ли OCR" — это дождаться
#   таймаута/повтора внутри самого OCR-запроса (до 45+70 сек). Теперь:
#   отдельная лёгкая ручка /api/ocr-health с кэшем, который фоновый
#   поток обновляет каждые OCR_HEALTH_PING_INTERVAL секунд — заодно
#   это и есть keep-alive пинг, чтобы Render-донор не засыпал.
# ============================================================
OCR_REMOTE_HEALTH_URL = os.environ.get(
    "OCR_REMOTE_HEALTH_URL",
    OCR_REMOTE_URL.rsplit("/", 1)[0] + "/health" if OCR_REMOTE_URL else "",
)
OCR_HEALTH_PING_INTERVAL = float(os.environ.get("OCR_HEALTH_PING_INTERVAL", 600))  # 10 мин

_ocr_health_cache = {"ok": None, "checked_at": 0.0, "detail": None}
_ocr_health_lock = threading.Lock()


def _refresh_ocr_health(timeout: float = 6.0):
    ok, detail = None, None
    try:
        r = requests.get(OCR_REMOTE_HEALTH_URL, timeout=timeout)
        ok = r.status_code == 200
        if ok:
            try:
                detail = r.json()
            except ValueError:
                detail = None
    except Exception as e:
        ok, detail = False, str(e)
    with _ocr_health_lock:
        _ocr_health_cache.update({"ok": ok, "checked_at": time.monotonic(), "detail": detail})
    print(f"[ocr_health] {'OK' if ok else 'НЕДОСТУПЕН'}")
    return dict(_ocr_health_cache)


def _ocr_keepalive_loop():
    while True:
        try:
            _refresh_ocr_health()
        except Exception as e:
            print(f"[ocr_health] ошибка фонового пинга: {e}")
        time.sleep(OCR_HEALTH_PING_INTERVAL)


if OCR_REMOTE_HEALTH_URL:
    threading.Thread(target=_ocr_keepalive_loop, daemon=True).start()


@app.get("/api/ocr-health")
async def api_ocr_health():
    """Мгновенный ответ из кэша — не ждём и не будим сервис синхронно.
    Если кэш ещё вообще пустой (первый запрос после деплоя), даём фронту
    честное 'unknown', а фоновый поток обновит его в ближайшие секунды."""
    with _ocr_health_lock:
        cache = dict(_ocr_health_cache)
    if cache["checked_at"] == 0.0:
        return {"ok": None, "checked_seconds_ago": None, "detail": None}
    return {
        "ok": cache["ok"],
        "checked_seconds_ago": round(time.monotonic() - cache["checked_at"], 1),
        "detail": cache["detail"],
    }


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


class RecipeThumbItem(BaseModel):
    id: object  # int (recipe id) — object чтобы не падать, если фронт когда-нибудь пришлёт строку
    params: dict


class ApplyRecipesBatchRequest(BaseModel):
    url: str
    recipes: list[RecipeThumbItem]
    max_side: Optional[int] = 240  # уменьшаем превью перед кодированием — это просто миниатюра для сетки


def _shrink_for_thumb(out: np.ndarray, max_side: int) -> np.ndarray:
    if max_side and max_side > 0:
        h, w = out.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / float(longest)
            out = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out


# Тот же ресайз, что и для превьюшек, но отдельная константа под OCR:
# нужен, чтобы падало не только на "Оригинал (без обработки)" — рецепт
# raw/без изменений отдаёт картинку в исходном разрешении с телефона,
# и именно она (а не сами фильтры) раздувала base64-пейлоад на весь
# батч и роняла запрос по таймауту на 0.1 vCPU Render. Применяем ПЕРЕД
# отправкой в OCR всегда, независимо от того, что выбрал рецепт.
OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", 900))


def _cap_for_ocr(out: np.ndarray) -> np.ndarray:
    return _shrink_for_thumb(out, OCR_MAX_SIDE)


@app.post("/api/apply-recipes-batch")
async def api_apply_recipes_batch(payload: ApplyRecipesBatchRequest):
    """Как /api/apply-recipe, но для ОДНОГО фото сразу считает превью ПОД
    ВЕСЬ набор рецептов за один HTTP-запрос (картинка скачивается/декодируется
    один раз). Нужно для таблицы 'фото × все фильтры' (аналог экспортированного
    экселя, но живьём в браузере) — без этого пришлось бы делать по одному
    запросу на каждую ячейку таблицы, что на 0.1 vCPU Render убило бы страницу."""
    try:
        bgr = fetch_image_bgr(payload.url)
        if bgr is None:
            return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Не удалось скачать фото: {e}"})

    results = []
    for item in payload.recipes:
        try:
            out = apply_recipe(bgr, item.params)
            out = _shrink_for_thumb(out, payload.max_side or 240)
            results.append({"id": item.id, "image": encode_png_base64(out)})
        except Exception as e:
            results.append({"id": item.id, "error": str(e)})
    return {"results": results, "width": bgr.shape[1], "height": bgr.shape[0]}


class OcrBatchItem(BaseModel):
    id: object  # id строки (int) — object чтобы не падать на нестандартном типе
    url: str


class OcrCheckFilterBatchRequest(BaseModel):
    params: dict
    items: list[OcrBatchItem]


# ИЗМЕНЕНИЕ: сначала пробовали батчить фильтры ОДНОЙ строки в один
# contact-sheet-запрос — Gemini путается, т.к. все ячейки такого листа
# визуально ОДИН И ТОТ ЖЕ чек, и он копирует один найденный ответ на
# соседние нечитаемые ячейки вместо честного null (см. скрин: "872"
# на визуально разных фильтрах и даже между разными чеками). Раз ответ
# не "0" — внутренний retry/dddd-фолбэк донора не срабатывает.
# Правильный батч — наоборот: ОДИН фильтр × МНОГО РАЗНЫХ чеков (реально
# разные картинки, как в вашем ocr_hard-тесте на 5352 капчах, где
# путаницы не было). Именно так собирает пачку это эндпоинт — вызывается
# из bulkRecognize с внешним циклом по фильтрам.
OCR_CROSS_ROW_BATCH_CHUNK = int(os.environ.get("OCR_CROSS_ROW_BATCH_CHUNK", 40))


@app.post("/api/ocr-check-filter-batch")
async def api_ocr_check_filter_batch(payload: OcrCheckFilterBatchRequest):
    """Один фильтр (params) применяется к N РАЗНЫМ чекам (разные url —
    реально разные цифры), результаты батчатся в OCR чанками по
    OCR_CROSS_ROW_BATCH_CHUNK и шлются параллельно. Возвращает список
    {id, digits, source, error} в порядке payload.items."""
    if not payload.items:
        return JSONResponse(content={"results": []}, headers={"Cache-Control": "no-store"})

    images_bytes = []
    ok_ids = []
    results_by_id = {}
    for item in payload.items:
        try:
            bgr = fetch_image_bgr(item.url)
            if bgr is None:
                results_by_id[item.id] = {"id": item.id, "digits": None, "source": None, "error": "не удалось декодировать изображение"}
                continue
            out = apply_recipe(bgr, payload.params)
            out = _cap_for_ocr(out)
            images_bytes.append(encode_png_bytes(out))
            ok_ids.append(item.id)
        except Exception as e:
            results_by_id[item.id] = {"id": item.id, "digits": None, "source": None, "error": str(e)}

    chunk_size = max(1, OCR_CROSS_ROW_BATCH_CHUNK)
    id_chunks = [ok_ids[i:i + chunk_size] for i in range(0, len(ok_ids), chunk_size)]
    img_chunks = [images_bytes[i:i + chunk_size] for i in range(0, len(images_bytes), chunk_size)]

    chunk_results = await asyncio.gather(*[
        asyncio.to_thread(extract_via_remote_ocr_batch, chunk) for chunk in img_chunks
    ])

    for ids_part, results_part in zip(id_chunks, chunk_results):
        for rid, (digits, source, err) in zip(ids_part, results_part):
            results_by_id[rid] = {"id": rid, "digits": digits, "source": source, "error": err}

    ordered = [results_by_id[item.id] for item in payload.items]
    return JSONResponse(content={"results": ordered}, headers={"Cache-Control": "no-store"})


# ============================================================
#         ФОНОВЫЙ JOB ДЛЯ /api/ocr-check-filter-batch
#
#   Раньше весь прогон "по каждому фильтру — батч по ещё не найденным
#   чекам" жил ЦЕЛИКОМ в JS-цикле bulkRecognize() во вкладке браузера:
#   состояние (pending/matched, номер текущего фильтра) — в обычных
#   JS-переменных. Закрыли вкладку, обновили страницу, легла сеть или
#   раздутый "Оригинал" уронил один fetch без внятного catch — и весь
#   прогресс терялся, начинай сначала.
#
#   Теперь job живёт в таблице ocr_jobs (см. 04_ocr_jobs.sql), а крутит
#   его фоновый поток ВНУТРИ этого же процесса — без внешнего крона,
#   ровно тот же приём, что и _ocr_keepalive_loop ниже. На редеплое
#   Render поток умирает вместе с процессом, но состояние — в БД, а не
#   в памяти, поэтому при старте процесса мы просто подхватываем job,
#   если он остался в статусе running/stopping (см. _resume_ocr_jobs()
#   в конце файла). Одновременно может идти только один job — это
#   внутренний инструмент на одного оператора, не веб-сервис на разных
#   пользователей, поэтому очередь из нескольких job'ов не нужна:
#   уникальный частичный индекс в БД (см. SQL) просто не даст создать
#   второй, пока первый running/stopping.
# ============================================================
class StartOcrJobRequest(BaseModel):
    row_ids: list
    recipes: list[dict]  # [{id, name, params}, ...] — порядок = порядок попыток


def _ocr_job_pending_rows(row_ids: list) -> dict:
    """id -> {url, label} для строк, которые ЕЩЁ не reviewed (не найден
    рабочий фильтр). Источник истины по прогрессу — сама price_checks,
    а не job: так поллинг всегда честный, даже если оператор в процессе
    job'а вручную разметил строку из другого места."""
    result = (supabase.table("price_checks")
              .select("id,url,label,reviewed")
              .in_("id", row_ids)
              .execute())
    return {r["id"]: r for r in result.data if not r.get("reviewed")}


def _ocr_job_status(job_id: int) -> Optional[str]:
    row = supabase.table("ocr_jobs").select("status").eq("id", job_id).maybe_single().execute().data
    return row["status"] if row else None


def _run_ocr_job(job_id: int):
    try:
        job = supabase.table("ocr_jobs").select("*").eq("id", job_id).single().execute().data
    except Exception as e:
        print(f"[ocr_job {job_id}] не удалось прочитать job: {e}")
        return

    row_ids = job["row_ids"]
    recipes = job["recipes"]
    results = job.get("results") or {}
    chunk_size = max(1, OCR_CROSS_ROW_BATCH_CHUNK)

    print(f"[ocr_job {job_id}] старт: {len(row_ids)} строк, {len(recipes)} фильтров, с фильтра #{job['current_recipe_idx']}")

    for ri in range(job["current_recipe_idx"], len(recipes)):
        pending = _ocr_job_pending_rows(row_ids)
        if not pending:
            break
        if _ocr_job_status(job_id) == "stopping":
            supabase.table("ocr_jobs").update({"status": "stopped"}).eq("id", job_id).execute()
            print(f"[ocr_job {job_id}] остановлен оператором на фильтре #{ri}")
            return

        recipe = recipes[ri]
        pending_ids = list(pending.keys())
        id_chunks = [pending_ids[i:i + chunk_size] for i in range(0, len(pending_ids), chunk_size)]

        for id_chunk in id_chunks:
            if _ocr_job_status(job_id) == "stopping":
                supabase.table("ocr_jobs").update({"status": "stopped"}).eq("id", job_id).execute()
                print(f"[ocr_job {job_id}] остановлен оператором внутри фильтра #{ri}")
                return

            images_bytes = []
            ok_ids = []
            for rid in id_chunk:
                try:
                    bgr = fetch_image_bgr(pending[rid]["url"])
                    if bgr is None:
                        continue
                    out = apply_recipe(bgr, recipe["params"])
                    out = _cap_for_ocr(out)
                    images_bytes.append(encode_png_bytes(out))
                    ok_ids.append(rid)
                except Exception as e:
                    print(f"[ocr_job {job_id}] чек #{rid} — ошибка обработки: {e}")

            chunk_results = extract_via_remote_ocr_batch(images_bytes)
            for rid, (digits, source, err) in zip(ok_ids, chunk_results):
                label = str(pending[rid].get("label") or "")
                is_match = bool(digits and label and digits.lstrip("0") == label.lstrip("0"))
                results.setdefault(str(rid), {})[str(recipe["id"])] = {
                    "digits": digits, "match": is_match, "error": err,
                }
                if is_match:
                    supabase.table("price_checks").update({
                        "best_recipe_id": recipe["id"], "reviewed": True,
                    }).eq("id", rid).execute()

            supabase.table("ocr_jobs").update({"results": results}).eq("id", job_id).execute()

        supabase.table("ocr_jobs").update({"current_recipe_idx": ri + 1}).eq("id", job_id).execute()

    supabase.table("ocr_jobs").update({"status": "done"}).eq("id", job_id).execute()
    print(f"[ocr_job {job_id}] готово")


@app.post("/api/ocr-jobs")
async def start_ocr_job(payload: StartOcrJobRequest):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    if not payload.row_ids:
        return JSONResponse(status_code=400, content={"error": "Нужны row_ids"})

    try:
        job = supabase.table("ocr_jobs").insert({
            "row_ids": payload.row_ids,
            "recipes": payload.recipes,
        }).execute().data[0]
    except Exception:
        # уникальный индекс на "только один активный job" не дал вставить —
        # значит, job уже идёт; отдаём его фронту вместо ошибки.
        active = (supabase.table("ocr_jobs").select("*")
                  .in_("status", ["running", "stopping"])
                  .order("id", desc=True).limit(1).execute().data)
        if active:
            return active[0]
        return JSONResponse(status_code=500, content={"error": "Не удалось создать job"})

    threading.Thread(target=_run_ocr_job, args=(job["id"],), daemon=True).start()
    return job


@app.post("/api/ocr-jobs/{job_id}/stop")
async def stop_ocr_job(job_id: int):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    supabase.table("ocr_jobs").update({"status": "stopping"}).eq("id", job_id).execute()
    return {"message": "ok"}


@app.get("/api/ocr-jobs/active")
async def get_active_ocr_job():
    """Фронт дёргает это при загрузке /broken — если job уже идёт (в т.ч.
    после редеплоя, пока вы читали лог), прогресс-бар и ячейки оживают
    сами, без повторного нажатия кнопки."""
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    rows = (supabase.table("ocr_jobs").select("*")
            .in_("status", ["running", "stopping"])
            .order("id", desc=True).limit(1).execute().data)
    return {"job": rows[0] if rows else None}


@app.get("/api/ocr-jobs/{job_id}")
async def get_ocr_job(job_id: int):
    try:
        require_supabase()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    row = supabase.table("ocr_jobs").select("*").eq("id", job_id).maybe_single().execute().data
    if not row:
        return JSONResponse(status_code=404, content={"error": "Job не найден"})
    return {"job": row}


def _resume_ocr_jobs():
    """Вызывается один раз при старте процесса. Если процесс перезапустился
    (редеплой, падение по памяти, сон/пробуждение на Render) посреди
    job'а — статус в БД остался running/stopping, а поток, который его
    крутил, умер вместе со старым процессом. Поднимаем новый поток с того
    же current_recipe_idx, ничего не пересчитывая с нуля."""
    if supabase is None:
        return
    try:
        rows = (supabase.table("ocr_jobs").select("*")
                .in_("status", ["running", "stopping"]).execute().data)
    except Exception as e:
        print(f"[ocr_job] не удалось проверить незавершённые job'ы: {e}")
        return
    for job in rows:
        print(f"[ocr_job {job['id']}] подхватываю после рестарта процесса (статус={job['status']})")
        threading.Thread(target=_run_ocr_job, args=(job["id"],), daemon=True).start()


_resume_ocr_jobs()


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
