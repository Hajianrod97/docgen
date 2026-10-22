FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi uvicorn
COPY backend ./backend
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0"]
