import os
import io
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

app = FastAPI()

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Храним данные
current_data = []

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# МАРШРУТ ДЛЯ КАРТИНОК (обходит CORS)
@app.get("/proxy-image")
async def proxy_image(url: str):
    try:
        # Скачиваем картинку с Backblaze
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return Response(content=response.content, media_type="image/jpeg")
        else:
            return JSONResponse(status_code=404, content={"error": "Фото не найдено"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Загрузка CSV
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
        # Находим колонки автоматически
        url_col = 'test' if 'test' in df.columns else df.columns[0]
        label_col = 'label' if 'label' in df.columns else df.columns[1]
        
        current_data.append({
            "id": idx,
            "url": str(row[url_col]),
            "label": str(row[label_col]),
            "edited": str(row[label_col])
        })
    
    return {"message": f"Загружено {len(current_data)} строк"}

# Получить данные
@app.get("/api/data")
async def get_data():
    return current_data

# Сохранить данные
@app.post("/api/save")
async def save_data(new_data: list):
    global current_data
    current_data = new_data
    return {"message": "Данные сохранены"}

# Скачать итоговый CSV
@app.get("/download-csv")
async def download_csv():
    if not current_data:
        return JSONResponse(status_code=404, content={"error": "Нет данных"})
    
    # Создаем DataFrame
    df = pd.DataFrame(current_data)
    
    # Сохраняем в буфер
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=updated_prices.csv"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
