# Use lightweight Python base
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, model, and frontend template
COPY app.py .
COPY model.pkl .
COPY templates/ ./templates/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]