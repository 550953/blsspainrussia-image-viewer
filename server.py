# server.py
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io

from ocr_engine import extract_price_from_url

app = FastAPI()

# Обслуживаем статические файлы (ваш HTML)
app.mount("/static", StaticFiles(directory="."), name="static")

# Храним текущее состояние
current_data = []

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Отдаем ваш HTML
    with open("dash.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    global current_data
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    current_data = []
    for idx, row in df.iterrows():
        url = str(row['test'])
        label = str(row['label'])
        current_data.append({
            "id": idx,
            "url": url,
            "label": label,
            "ocr_result": None,
            "error": None
        })
    return {"message": f"Загружено {len(current_data)} строк"}

@app.post("/process-all/")
async def process_all():
    """Запускает OCR по всем загруженным картинкам."""
    for item in current_data:
        if item["error"] is None:
            val, err = extract_price_from_url(item["url"])
            item["ocr_result"] = val
            item["error"] = err
    return {"message": "Обработка завершена"}

@app.post("/process-one/{item_id}/")
async def process_one(item_id: int):
    item = current_data[item_id]
    val, err = extract_price_from_url(item["url"])
    item["ocr_result"] = val
    item["error"] = err
    return item

@app.get("/api/data/")
async def get_data():
    return current_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
