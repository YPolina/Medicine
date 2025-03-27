FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY utils.py .
COPY analysis.py .

RUN apt update && apt install -y libxrender1
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]