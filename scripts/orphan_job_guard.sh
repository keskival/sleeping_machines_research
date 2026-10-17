#!/usr/bin/env bash
# Replacement watchdog for a run_safe job whose runner shell died (Curie, 17 Oct: the runner was killed with its tmux session
# while the timeout/python pair kept running). Same limits as the chain's run_safe call: process RSS cap, MemAvailable floor,
# cgroup headroom. On breach: TERM the timeout process (it forwards to python; the job resumes exactly from its last checkpoint).
# Usage: orphan_job_guard.sh <timeout pid> <python pid> <log>
T=$1; P=$2; LOG=$3; RSS_CAP=${MEM_CAP_RSS_KB:-3000000}; MIN_MB=${MIN_AVAIL_MB:-8192}; RES_MB=${CGROUP_RESERVE_MB:-2048}
while kill -0 "$P" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print $2}' /proc/$P/status 2>/dev/null || echo 0); avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo)/1024 ))
  mx=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  head=999999; [ -n "$mx" ] && [ "$mx" != max ] && head=$(( (mx - cur)/1048576 ))
  if [ "${rss:-0}" -gt "$RSS_CAP" ] || [ "$avail" -lt "$MIN_MB" ] || [ "$head" -lt "$RES_MB" ]; then
    echo "$(date +%T) GUARD STOP: RSS=${rss}KB MemAvailable=${avail}MB cgroup headroom=${head}MB" >> "$LOG"; kill -TERM "$T" 2>/dev/null; sleep 15; kill -KILL "$T" "$P" 2>/dev/null; exit 1
  fi
  sleep 2
done
echo "$(date +%T) guard: job exited" >> "$LOG"
