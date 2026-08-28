#!/bin/bash
# Managed by Ansible (roles/common/tasks/node_exporter_textfile.yml).
#
# Stand-in for node_exporter's built-in cpufreq collector, which is disabled
# cluster-wide (--no-collector.cpufreq). On the Grace CPUs the scaling driver is
# cppc_cpufreq, where reading cpuinfo_avg_freq blocks forever on an idle core
# (the PCC firmware mailbox never answers). node_exporter reads every policy in
# parallel through an errgroup and waits on the WaitGroup, so that ONE stuck
# read wedges the entire scrape and leaks a goroutine plus an fd per attempt.
# Upstream: node_exporter#3791, fix proposed in procfs#861 (both open on
# 2026-08-28). Measured here 2026-08-28: cpuinfo_avg_freq hung on 4 of 20 CPUs
# in one pass and on 6 other CPUs in the next; every other cpufreq attribute
# read fine on all 20.
#
# So this script reads the SAME attributes minus cpuinfo_avg_freq, which is the
# only value that cannot be obtained at all.
#
# Reads are plain shell redirections, NOT `timeout cat`: bounding each of the
# ~7*ncpu reads individually costs two forks apiece (23 s wall clock for one
# pass when measured that way) and buys nothing, because a sysfs read stuck in
# the firmware mailbox is uninterruptible anyway. The bound that does work sits
# in the unit file: Type=oneshot with TimeoutStartSec, so a wedged pass is
# killed and the previous .prom simply stays in place as stale data.
#
# Metric names deliberately match upstream's cpufreq collector so existing
# dashboards (e.g. Grafana "Node Exporter Full") keep working. That is also why
# this must NOT run while the built-in collector is enabled: two sources for the
# same series fail the whole gather.

set -u

OUT_DIR="${1:-/var/lib/node_exporter/textfile}"
OUT_FILE="${OUT_DIR}/cpufreq.prom"
CPU_BASE=/sys/devices/system/cpu

# Fork-free read; empty/unreadable attribute yields rc 1.
read_attr() {
    local f="$1" v
    [ -r "$f" ] || return 1
    v="$(<"$f")" || return 1
    [ -n "$v" ] || return 1
    printf '%s' "$v"
}

# kHz (sysfs unit) -> Hz, without relying on bc/awk for the multiply.
khz_to_hz() {
    printf '%s' "$(( $1 * 1000 ))"
}

emit_freq() {
    local metric="$1" cpu="$2" file="$3" raw
    raw="$(read_attr "$file")" || return 0
    case "$raw" in
        ''|*[!0-9]*) return 0 ;;   # skip "<unknown>" and friends
    esac
    printf '%s{cpu="%s"} %s\n' "$metric" "$cpu" "$(khz_to_hz "$raw")"
}

TMP="$(mktemp "${OUT_DIR}/.cpufreq.prom.XXXXXX")" || exit 1
trap 'rm -f "$TMP"' EXIT

{
    echo '# HELP node_cpu_scaling_frequency_hertz Current scaled CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_scaling_frequency_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_scaling_frequency_hertz "$cpu" "$d/scaling_cur_freq"
    done

    echo '# HELP node_cpu_scaling_frequency_max_hertz Maximum scaled CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_scaling_frequency_max_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_scaling_frequency_max_hertz "$cpu" "$d/scaling_max_freq"
    done

    echo '# HELP node_cpu_scaling_frequency_min_hertz Minimum scaled CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_scaling_frequency_min_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_scaling_frequency_min_hertz "$cpu" "$d/scaling_min_freq"
    done

    # cpuinfo_* is the hardware's own view. Present on the Grace nodes
    # (cppc_cpufreq), absent for cpuinfo_cur_freq on intel_pstate — emit_freq
    # simply skips what does not exist.
    echo '# HELP node_cpu_frequency_hertz Current CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_frequency_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_frequency_hertz "$cpu" "$d/cpuinfo_cur_freq"
    done

    echo '# HELP node_cpu_frequency_max_hertz Maximum CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_frequency_max_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_frequency_max_hertz "$cpu" "$d/cpuinfo_max_freq"
    done

    echo '# HELP node_cpu_frequency_min_hertz Minimum CPU thread frequency in hertz.'
    echo '# TYPE node_cpu_frequency_min_hertz gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        emit_freq node_cpu_frequency_min_hertz "$cpu" "$d/cpuinfo_min_freq"
    done

    # One series per (cpu, governor) pair with value 1 on the active governor,
    # exactly like upstream's collector.
    echo '# HELP node_cpu_scaling_governor Current enabled CPU frequency governor.'
    echo '# TYPE node_cpu_scaling_governor gauge'
    for d in "$CPU_BASE"/cpu[0-9]*/cpufreq; do
        [ -d "$d" ] || continue
        cpu="${d%/cpufreq}"; cpu="${cpu##*/cpu}"
        current="$(read_attr "$d/scaling_governor")" || continue
        available="$(read_attr "$d/scaling_available_governors")" || available="$current"
        for g in $available; do
            if [ "$g" = "$current" ]; then val=1; else val=0; fi
            printf 'node_cpu_scaling_governor{cpu="%s",governor="%s"} %s\n' "$cpu" "$g" "$val"
        done
    done
} > "$TMP"

chmod 0644 "$TMP"
# Atomic swap: the collector never sees a half-written file.
mv -f "$TMP" "$OUT_FILE"
trap - EXIT
