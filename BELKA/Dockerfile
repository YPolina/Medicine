FROM python:3.10.12

WORKDIR /app

COPY requirements.txt .
COPY xgb_cls.pkl .
COPY functions,py .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]


