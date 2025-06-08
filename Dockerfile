FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Install both requirement files
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r torch-requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]