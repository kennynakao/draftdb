# DraftDB trainer: the lab engine with TabICL on CUDA. Built in ACR (linux/amd64), run as a Container Apps Job on one A100.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt
COPY bundle/ /workspace/
RUN useradd -m -u 10001 trainer && chown -R trainer:trainer /workspace
USER trainer
ENV HOME=/home/trainer PYTHONUNBUFFERED=1 TABICL_DEVICE=cuda DRAFTDB_LAB_OUT=/workspace/out
# pre-download the TabICL checkpoint so the job needs no network at run time
RUN python -c "import numpy as np; from tabicl import TabICLRegressor; m=TabICLRegressor(n_estimators=1, device='cpu'); m.fit(np.random.rand(60,3), np.random.rand(60)); m.predict(np.random.rand(2,3)); print('tabicl checkpoint cached')"
CMD ["python", "/workspace/train.py"]
