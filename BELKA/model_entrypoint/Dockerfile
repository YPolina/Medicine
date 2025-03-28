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

RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Palina123/HSA_MolFormerXL.ckpt', filename='HSA_MolFormer.ckpt', local_dir='/trained_models'); \
hf_hub_download(repo_id='Palina123/sEH_MolFormer.ckpt', filename='sEH_MolFormer.ckpt', local_dir='/trained_models'); \
hf_hub_download(repo_id='Palina123/HSA_ChemBert.ckpt', filename='HSA_ChemBert.ckpt', local_dir='/trained_models'); \
hf_hub_download(repo_id='Palina123/sEH_ChemBert.ckpt', filename='sEH_ChemBert.ckpt', local_dir='/trained_models'); \
hf_hub_download(repo_id='Palina123/BRD4_MolFormerXL.ckpt', filename='BRD4_MolFormer.ckpt', local_dir='/trained_models'); \
hf_hub_download(repo_id='Palina123/BRD4_ChemBert.ckpt', filename='BRD4_ChemBert.ckpt', local_dir='/trained_models')"

ENTRYPOINT ["python3", "inference.py"]
