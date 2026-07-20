FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

RUN python -m pip install --no-cache-dir \
    transformers==4.52.4 datasets==4.0.0 accelerate==1.9.0 \
    scipy==1.15.3 wandb==0.21.0 pytest==8.4.1 pyyaml==6.0.2 \
    jsonlines==4.0.0 openai==1.99.9 math-verify==0.8.0 \
    latex2sympy2-extended==1.10.2

WORKDIR /opt/fpct
COPY . /opt/fpct
ARG FPCT_GIT_SHA
ARG FPCT_GIT_BRANCH=research/fpct-factorized-transport
ARG FPCT_GIT_UPSTREAM
RUN python /opt/fpct/script/runtime/fpct_image_provenance.py \
    --root /opt/fpct --head "${FPCT_GIT_SHA}" --branch "${FPCT_GIT_BRANCH}" \
    --upstream "${FPCT_GIT_UPSTREAM}"
ENV PYTHONNOUSERSITE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
