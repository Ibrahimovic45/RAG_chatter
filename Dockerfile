# app/Dockerfile

FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel 

WORKDIR /RAG_chatter-main

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . ./

RUN pip3 install --default-timeout=100 -r requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN export MAX_JOBS=32
RUN pip3 install flash_attn==2.6.3 --no-build-isolation

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "rag_chatbot.py", "--server.port=8080", "--server.address=0.0.0.0"]

