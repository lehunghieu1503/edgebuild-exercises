#!/usr/bin/env bash
# Đưa máy vào / ra khỏi chế độ đo.
#
#   bench_mode.sh on      governor performance, turbo theo TURBO=0|1 (mặc định 0 = tắt turbo)
#   bench_mode.sh off     khôi phục trạng thái đã lưu lúc on
#   bench_mode.sh status  in trạng thái; exit 0 chỉ khi máy đang ở chế độ đo
#
# Biến môi trường:
#   TURBO=0|1          chọn một chế độ turbo và ghi vào mọi kết quả (mặc định 0)
#   ALLOW_BATTERY=1    cho phép on khi đang chạy pin (chỉ dùng khi cố ý đo chế độ pin)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/bench-kit"
STATE="$STATE_DIR/bench_mode.state"
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
TURBO="${TURBO:-0}"

[[ -r "$ROOT/cores.env" ]] && source "$ROOT/cores.env"
BENCH_CORE="${BENCH_CORE:-2}"

write_sys() {  # write_sys <giá trị> <file>
    if [[ -w "$2" ]]; then echo "$1" > "$2"; else echo "$1" | sudo tee "$2" > /dev/null; fi
}

governors() { cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | sort | uniq -c | awk '{printf "%s%s x%s", sep, $2, $1; sep=", "} END {print ""}'; }

# In "ac" nếu có ít nhất một nguồn ngoài (Mains/USB) đang online, ngược lại "battery".
power_source() {
    local f
    for f in /sys/class/power_supply/*/online; do
        [[ -r "$f" && "$(cat "$f")" == "1" ]] && { echo ac; return; }
    done
    echo battery
}

# Không dùng `cat | grep -q`: grep thoát sớm làm cat nhận SIGPIPE, pipefail biến thành lỗi giả.
all_performance() { ! grep -qvx performance /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; }

print_status() {
    echo "governor: $(governors)"
    echo "no_turbo: $(cat "$NO_TURBO") (turbo $([[ "$(cat "$NO_TURBO")" == 0 ]] && echo bật || echo tắt))"
    echo "nguồn:    $(power_source)"
    echo "state:    $([[ -f "$STATE" ]] && echo "đã lưu ở $STATE" || echo "chưa lưu (bench mode chưa bật bằng script này)")"
}

cmd_on() {
    [[ "$TURBO" == 0 || "$TURBO" == 1 ]] || { echo "TURBO phải là 0 hoặc 1" >&2; exit 2; }
    if [[ "$(power_source)" == battery && "${ALLOW_BATTERY:-0}" != 1 ]]; then
        echo "từ chối: máy đang chạy pin. Cắm sạc, hoặc đặt ALLOW_BATTERY=1 nếu cố ý đo chế độ pin." >&2
        exit 1
    fi

    # Chỉ lưu lần đầu: gọi on hai lần liên tiếp không được ghi đè trạng thái gốc.
    if [[ ! -f "$STATE" ]]; then
        mkdir -p "$STATE_DIR"
        {
            echo "no_turbo $(cat "$NO_TURBO")"
            for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
                echo "governor $f $(cat "$f")"
            done
        } > "$STATE"
    fi

    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        write_sys performance "$f"
    done
    write_sys "$((1 - TURBO))" "$NO_TURBO"

    all_performance || { echo "lỗi: không đặt được governor performance cho mọi CPU" >&2; exit 1; }
    print_status
    echo
    echo "pin benchmark vào một P-core (không dùng CPU 0):"
    echo "  taskset -c $BENCH_CORE <lệnh>"
}

cmd_off() {
    [[ -f "$STATE" ]] || { echo "không có trạng thái đã lưu ($STATE); không đổi gì." >&2; exit 1; }
    local kind a b
    while read -r kind a b; do
        case "$kind" in
            no_turbo) write_sys "$a" "$NO_TURBO" ;;
            governor) write_sys "$b" "$a" ;;
        esac
    done < "$STATE"
    rm -f "$STATE"
    print_status
}

cmd_status() {
    print_status
    local ok=0
    all_performance || { echo "CHƯA ở chế độ đo: governor chưa phải performance" >&2; ok=1; }
    if [[ "$(power_source)" == battery && "${ALLOW_BATTERY:-0}" != 1 ]]; then
        echo "CHƯA ở chế độ đo: đang chạy pin" >&2; ok=1
    fi
    return "$ok"
}

case "${1:-}" in
    on) cmd_on ;;
    off) cmd_off ;;
    status) cmd_status ;;
    *) echo "dùng: $0 on|off|status   (TURBO=0|1, ALLOW_BATTERY=1)" >&2; exit 2 ;;
esac
