#!/usr/bin/env bash
# Noise test của giàn đo: chạy cùng một micro-benchmark RUNS lần (mỗi lần một process riêng),
# in (max p50 - min p50) / median p50. Exit != 0 khi vượt THRESHOLD_PCT.
#
#   noise.sh <đường dẫn tới saxpy>
#
# Biến môi trường: RUNS (10), THRESHOLD_PCT (2), BENCH_CORE (lấy từ cores.env).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cores.env"
BIN="${1:?dùng: noise.sh <saxpy>}"
RUNS="${RUNS:-10}"
THRESHOLD_PCT="${THRESHOLD_PCT:-2}"

if ! "$ROOT/scripts/bench_mode.sh" status > /dev/null 2>&1; then
    echo "cảnh báo: máy chưa ở chế độ đo (chạy scripts/bench_mode.sh on)" >&2
fi

p50s=()
for ((i = 1; i <= RUNS; i++)); do
    line="$(taskset -c "$BENCH_CORE" "$BIN")"
    p50="$(sed -n 's/.*p50_ns=\([0-9.]*\).*/\1/p' <<< "$line")"
    [[ -n "$p50" ]] || { echo "không đọc được p50 từ: $line" >&2; exit 1; }
    printf 'run %2d: p50 = %s ns\n' "$i" "$p50"
    p50s+=("$p50")
done

printf '%s\n' "${p50s[@]}" | sort -n | awk -v thr="$THRESHOLD_PCT" -v core="$BENCH_CORE" '
    { v[NR] = $1 }
    END {
        n = NR
        median = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
        spread = (v[n] - v[1]) / median * 100
        printf "core %s, %d lần: min %.0f, median %.0f, max %.0f ns\n", core, n, v[1], median, v[n]
        printf "noise = (max - min) / median = %.2f %% (ngưỡng %s %%)\n", spread, thr
        if (spread > thr) { print "FAIL"; exit 1 }
        print "PASS"
    }'
