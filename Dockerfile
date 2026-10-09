FROM python:3.11-slim

WORKDIR /app

# 核心链路零第三方依赖，所以这里不装任何 requirements
COPY src/ ./src/
COPY app/ ./app/
COPY demo.py eval/ config.json ./
COPY data/ ./data/
COPY README.md ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 7860

CMD ["python", "-m", "app.api"]
