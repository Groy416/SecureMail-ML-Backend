#!/bin/sh
set -eu
mkdir -p /captures
tcpdump -U -i eth0 -w "${PCAP_PATH:-/captures/capture.pcap}" 'tcp port 25 or tcp port 143 or tcp port 110' &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' INT TERM EXIT
wait "$pid"
