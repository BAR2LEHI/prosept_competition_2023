FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

CMD ["uvicorn", "api.main:app", "--proxy-headers", "--host", "0.0.0.0 ", "--port", "8001"]
