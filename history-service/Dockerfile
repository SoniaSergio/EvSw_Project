FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crea i package marker vuoti se mancanti (robustezza)
RUN touch routers/__init__.py services/__init__.py schemas/__init__.py 2>/dev/null || true

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]