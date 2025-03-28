FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
COPY functionality/ .
COPY trained_models/ .
COPY inference.py .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "inference.py"]
