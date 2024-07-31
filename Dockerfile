# app/Dockerfile

FROM python:3.9-slim

WORKDIR /RAG_chatter-main

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . ./

RUN pip3 install -r requirements.txt

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "rag_chatbot.py", "--server.port=8080", "--server.address=0.0.0.0"]

