#!/bin/sh
set -eu

mkdir -p /captures
tcpdump -U -i eth0 -w "${PCAP_PATH:-/captures/capture.pcap}" 'tcp port 2525 or tcp port 1143' &
capture_pid=$!
python /app/server.py &
server_pid=$!

cleanup() {
  kill "$server_pid" "$capture_pid" 2>/dev/null || true
  wait "$server_pid" "$capture_pid" 2>/dev/null || true
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT
wait "$server_pid"
