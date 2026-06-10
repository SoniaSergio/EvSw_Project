FROM python:3.11-slim
WORKDIR /app
RUN pip install pymongo pandas cryptography
COPY seed_db.py .
CMD ["python", "seed_db.py"]