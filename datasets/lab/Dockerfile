FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --no-install-recommends -y openssl tcpdump tshark \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY entrypoint.sh capture.sh server.py certs.py client.py openssl-legacy.cnf /app/
RUN chmod +x /app/entrypoint.sh /app/capture.sh
