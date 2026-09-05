# Нужен ТОЛЬКО из-за OCR (/api/ocr-check, /api/ocr-scan).
#
# pytesseract — это Python-обёртка, ей нужен системный бинарник
# tesseract-ocr. На Render 'Native' environment apt-get недоступен
# (permissions restricted), поэтому единственный надёжный способ —
# Docker-деплой: при создании/редактировании сервиса на Render выбрать
# Environment = Docker (вместо Python), указать этот Dockerfile —
# автодеплой на каждый git push продолжит работать как раньше.
#
# Если OCR не нужен — можно продолжать деплоить как раньше (Python
# native environment), просто /api/ocr-check и /api/ocr-scan будут
# отвечать ошибкой "Tesseract не установлен" (см. /api/status),
# остальное приложение (фильтры, /broken и т.д.) работает без Docker.

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Render free tier = 0.1 общего vCPU. numpy/OpenCV/Tesseract по умолчанию
# норовят завести свой пул потоков — на одном слабом ядре это не ускоряет
# счёт, а только тратит время на переключение контекста и лишнюю память.
# (main.py дублирует это через os.environ.setdefault — на случай, если
# когда-нибудь уйдёте с Docker и эти ENV не будут выставлены платформой.)
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1

# Не обязателен для самого Render (у него свой health-check в настройках
# сервиса), но полезен при локальном "docker run"/docker-compose — не
# зря start-period большой: free-инстанс просыпается из сна 30-50 секунд.
HEALTHCHECK --interval=30s --timeout=5s --start-period=50s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=4)" || exit 1

CMD ["python", "main.py"]
