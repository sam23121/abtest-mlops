FROM jupyter/scipy-notebook

RUN mkdir my-model
ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib

RUN pip install joblib

COPY adSmartABdata.csv ./data/adSmartABdata.csv


COPY train.py ./scripts/train.py

RUN python3 train.py