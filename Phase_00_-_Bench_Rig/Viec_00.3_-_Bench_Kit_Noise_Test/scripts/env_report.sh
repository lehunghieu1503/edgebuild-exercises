#!/usr/bin/env bash
# In môi trường đo ra stdout dưới dạng JSON. Mọi con số benchmark phải đi kèm file này.
#
# Compiler và flags lấy từ biến môi trường BENCH_CXX / BENCH_CXXFLAGS (CMake truyền vào);
# không có thì dùng CXX / CXXFLAGS, rồi tới c++.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -r "$ROOT/cores.env" ]] && source "$ROOT/cores.env"

json_str() {  # chuỗi bất kỳ -> chuỗi JSON có ngoặc kép
    local s="$1"
    s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; s="${s//$'\n'/\\n}"; s="${s//$'\t'/\\t}"
    printf '"%s"' "$s"
}

cxx="${BENCH_CXX:-${CXX:-c++}}"
cxxflags="${BENCH_CXXFLAGS:-${CXXFLAGS:-}}"
cxx_version="$("$cxx" --version 2>/dev/null | head -n 1 || true)"

cpu_model="$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)"
governor="$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | sort -u | paste -sd, -)"
no_turbo="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo null)"

power=battery
for f in /sys/class/power_supply/*/online; do
    [[ -r "$f" && "$(cat "$f")" == "1" ]] && power=ac
done

commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)"
dirty=false
if [[ -n "$commit" && -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then dirty=true; fi

read -r load1 load5 load15 _ < /proc/loadavg

thermal=""
for z in /sys/class/thermal/thermal_zone*; do
    t="$(cat "$z/temp" 2>/dev/null || true)"
    [[ "$t" =~ ^-?[0-9]+$ ]] || continue
    thermal+="${thermal:+, }{\"zone\": $(json_str "$(cat "$z/type")"), \"celsius\": $(awk -v t="$t" 'BEGIN {printf "%.1f", t / 1000}')}"
done

cat <<JSON
{
  "timestamp": $(json_str "$(date -Is)"),
  "cpu_model": $(json_str "$cpu_model"),
  "p_cores": $(json_str "${P_CORES:-}"),
  "e_cores": $(json_str "${E_CORES:-}"),
  "governor": $(json_str "$governor"),
  "no_turbo": $no_turbo,
  "power": $(json_str "$power"),
  "thermal": [$thermal],
  "kernel": $(json_str "$(uname -r)"),
  "compiler": $(json_str "$cxx_version"),
  "cxxflags": $(json_str "$cxxflags"),
  "commit": $( [[ -n "$commit" ]] && json_str "$commit" || echo null ),
  "dirty": $dirty,
  "loadavg": [$load1, $load5, $load15]
}
JSON
