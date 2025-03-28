FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]