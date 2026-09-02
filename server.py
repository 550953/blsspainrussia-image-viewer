import os
import io
import base64
import requests
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_data = []


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ============================================================
#                    ЗАГРУЗКА КАРТИНКИ
# ============================================================
def fetch_image_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        content = fetch_image_bytes(url)
        return Response(content=content, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": f"Фото не найдено: {e}"})


# ============================================================
#              ФИЛЬТРЫ ДЛЯ ПРОВЕРКИ ЧИТАЕМОСТИ
# ============================================================
# Каждый фильтр: (key, label, func(bgr: np.ndarray) -> np.ndarray[gray or bgr])
# Работаем в основном в grayscale — так честнее оценивать читаемость текста.

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _auto_invert(gray: np.ndarray) -> np.ndarray:
    """Если фон светлый — инвертируем, чтобы текст стабильно был тёмным на светлом или наоборот."""
    return 255 - gray if float(np.mean(gray)) > 140 else gray


def f_original(bgr):
    return bgr


def f_fast_clahe(bgr):
    base = _auto_invert(_to_gray(bgr))
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    return clahe.apply(base)


def f_strong_contrast(bgr):
    base = _auto_invert(_to_gray(bgr))
    strong = cv2.convertScaleAbs(base, alpha=2.8, beta=-60)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    return clahe.apply(strong)


def f_adaptive_threshold(bgr):
    base = _auto_invert(_to_gray(bgr))
    binary = cv2.adaptiveThreshold(
        base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def f_aggressive_otsu(bgr):
    base = _auto_invert(_to_gray(bgr))
    den = cv2.GaussianBlur(base, (3, 3), 0)
    den = cv2.convertScaleAbs(den, alpha=3.2, beta=-80)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    den = clahe.apply(den)
    _, otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def f_rb_diff(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    val = (r - b) * 3.0 + 128
    return np.clip(val, 0, 255).astype(np.uint8)


def f_median_stretch(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    gray = r * 0.4 + g * 0.4 + b * 0.2
    gray = (gray - 180) * 4.5
    return np.clip(gray, 0, 255).astype(np.uint8)


def f_invert_contrast(bgr):
    gray = _to_gray(bgr).astype(np.float32)
    inv = 255 - gray
    val = (inv - 128) * 2.2 + 128
    return np.clip(val, 0, 255).astype(np.uint8)


def f_max_readable(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    val = (r - b) * 2.8 + (r - g) * 0.6
    val = (val - 10) * 1.8 + 90
    return np.clip(val, 0, 255).astype(np.uint8)


def f_unsharp(bgr):
    """Резкость — помогает при смазанном / низкочастотном тексте."""
    gray = _auto_invert(_to_gray(bgr))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 2.2, blurred, -1.2, 0)
    return sharp


def f_bilateral_denoise(bgr):
    """Убирает мелкий точечный / шумовой фон (как у файлов с точечным паттерном),
    сохраняя резкие границы штрихов текста, затем бинаризация."""
    gray = _auto_invert(_to_gray(bgr))
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def f_morph_gradient(bgr):
    """Морфологический градиент — выделяет контуры штрихов, часто хорошо
    отделяет текст от повторяющегося фонового паттерна."""
    gray = _auto_invert(_to_gray(bgr))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    grad = cv2.convertScaleAbs(grad, alpha=2.5, beta=0)
    return grad


def f_local_mean_threshold(bgr):
    """Локальная бинаризация большим окном — устойчива к неравномерной
    засветке/градиенту фона."""
    gray = _auto_invert(_to_gray(bgr))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5
    )
    return binary


def f_upscale_sharpen(bgr):
    """Апскейл x3 + резкость — полезно на маленьких капчах (< 150px)."""
    gray = _auto_invert(_to_gray(bgr))
    h, w = gray.shape
    big = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(big, (0, 0), sigmaX=2)
    sharp = cv2.addWeighted(big, 1.8, blurred, -0.8, 0)
    return sharp


FILTERS = [
    ("original", "Оригинал", f_original),
    ("fast_clahe", "CLAHE (быстрый)", f_fast_clahe),
    ("strong_contrast", "Сильный контраст", f_strong_contrast),
    ("adaptive_threshold", "Адаптивная бинаризация", f_adaptive_threshold),
    ("aggressive_otsu", "Агрессивный OTSU", f_aggressive_otsu),
    ("rb_diff", "R-B разница", f_rb_diff),
    ("median_stretch", "Median + Stretch", f_median_stretch),
    ("invert_contrast", "Инверсия + контраст", f_invert_contrast),
    ("max_readable", "Макс. читаемость", f_max_readable),
    ("unsharp", "Резкость (unsharp)", f_unsharp),
    ("bilateral_denoise", "Подавление точечного шума", f_bilateral_denoise),
    ("morph_gradient", "Морф. градиент (контуры)", f_morph_gradient),
    ("local_mean_threshold", "Локальный порог (яркость)", f_local_mean_threshold),
    ("upscale_sharpen", "Апскейл x3 + резкость", f_upscale_sharpen),
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
    """Возвращает все варианты обработки картинки как base64 PNG —
    без канвас-обработки на клиенте, без проблем с CORS."""
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

    return {"width": bgr.shape[1], "height": bgr.shape[0], "filters": results}


# ============================================================
#                       CSV / ДАННЫЕ
# ============================================================
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global current_data
    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Ошибка чтения CSV: {str(e)}"})

    current_data = []
    for idx, row in df.iterrows():
        url_col = 'test' if 'test' in df.columns else df.columns[0]
        label_col = 'label' if 'label' in df.columns else df.columns[1]

        current_data.append({
            "id": idx,
            "url": str(row[url_col]),
            "label": str(row[label_col]),
            "edited": str(row[label_col])
        })

    return {"message": f"Загружено {len(current_data)} строк"}


@app.get("/api/data")
async def get_data():
    return current_data


@app.post("/api/save")
async def save_data(new_data: list):
    global current_data
    current_data = new_data
    return {"message": "Данные сохранены"}


@app.get("/download-csv")
async def download_csv():
    if not current_data:
        return JSONResponse(status_code=404, content={"error": "Нет данных"})

    df = pd.DataFrame(current_data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)

    return StreamingResponse(
        buffer, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=updated_prices.csv"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
