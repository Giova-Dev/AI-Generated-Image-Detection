FROM python:3.11-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app/ ./app/
COPY src/utils.py ./src/utils.py
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 5000

CMD ["python", "app/app.py"]