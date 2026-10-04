#!/usr/bin/env bash
# Xác định P-core / E-core và đối chiếu với cores.env. Exit != 0 nếu lệch.
#
# Nguồn ưu tiên: /sys/devices/cpu_core/cpus và /sys/devices/cpu_atom/cpus (PMU hybrid).
# Kernel chưa nhận PMU hybrid của CPU (ví dụ 6.8 với Arrow Lake) thì hai file này không có;
# khi đó suy ra từ cache L2: P-core có L2 riêng, E-core dùng chung L2 theo cụm 4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cores.env"

# "0 1 2 5 6" -> "0-2,5-6"
to_ranges() {
    tr ' ' '\n' | sort -n | awk '
        NF { if (start == "") { start = prev = $1; next }
             if ($1 == prev + 1) { prev = $1; next }
             out = out (out ? "," : "") (start == prev ? start : start "-" prev); start = prev = $1 }
        END { if (start != "") out = out (out ? "," : "") (start == prev ? start : start "-" prev); print out }'
}

if [[ -r /sys/devices/cpu_core/cpus && -r /sys/devices/cpu_atom/cpus ]]; then
    method="sysfs cpu_core/cpu_atom"
    p="$(cat /sys/devices/cpu_core/cpus)"
    e="$(cat /sys/devices/cpu_atom/cpus)"
else
    method="cache L2 (không có /sys/devices/cpu_core, cpu_atom trên kernel $(uname -r))"
    p_list=() e_list=()
    for d in /sys/devices/system/cpu/cpu[0-9]*; do
        n="${d##*cpu}"
        shared="$(cat "$d/cache/index2/shared_cpu_list")"
        if [[ "$shared" == "$n" ]]; then p_list+=("$n"); else e_list+=("$n"); fi
    done
    p="$(echo "${p_list[*]:-}" | to_ranges)"
    e="$(echo "${e_list[*]:-}" | to_ranges)"
fi

echo "phương pháp: $method"
echo "đo được:   P_CORES=$p E_CORES=$e"
echo "cores.env: P_CORES=$P_CORES E_CORES=$E_CORES"
lscpu --extended=CPU,CORE,MAXMHZ | awk 'NR == 1 || $1 == 0 || $1 == 2 || $1 == 8 || $1 == 12'

if [[ "$p" == "$P_CORES" && "$e" == "$E_CORES" ]]; then
    echo "OK"
else
    echo "LỆCH: sửa cores.env" >&2
    exit 1
fi
