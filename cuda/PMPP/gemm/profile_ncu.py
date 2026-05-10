#!/usr/bin/env python3
"""Run Nsight Compute for PMPP SGEMM benchmarks and summarize details CSVs.

Typical use from the repository root:

    python3 cuda/PMPP/gemm/profile_ncu.py

Typical use from the build directory:

    python3 /path/to/repo/cuda/PMPP/gemm/profile_ncu.py --bench ./PMPP_gemm_bench

The script has three stages:

1. profile: run `ncu` and write one `.ncu-rep` per benchmark variant.
2. import: convert each `.ncu-rep` to a details CSV with `ncu --import`.
3. summarize: extract the small set of counters that are most useful for this
   GEMM learning path and write `summary.csv` plus `summary.md`.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SECTIONS = (
    "SpeedOfLight",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "SourceCounters",
)


@dataclass(frozen=True)
class Variant:
    name: str
    benchmark: str
    kernel_regex: str = "sgemm_tiled_thread_tile_kernel"


DEFAULT_VARIANTS = (
    Variant("2x2", "SGEMM/Tiled16_2x2/1024"),
    Variant("4x4", "SGEMM/Tiled16_4x4/1024"),
    Variant("8x8", "SGEMM/Tiled16_8x8/1024"),
    Variant("2x2K32", "SGEMM/Tiled16_2x2K32/1024"),
    Variant("2x2K64", "SGEMM/Tiled16_2x2K64/1024"),
    Variant("4x4K32", "SGEMM/Tiled16_4x4K32/1024"),
    Variant("4x4K64", "SGEMM/Tiled16_4x4K64/1024"),
    Variant("8x8K32", "SGEMM/Tiled16_8x8K32/1024"),
    Variant("8x8K64", "SGEMM/Tiled16_8x8K64/1024"),
)


SUMMARY_COLUMNS = (
    "variant",
    "duration_us",
    "derived_tflops",
    "compute_sm_pct",
    "memory_sol_pct",
    "dram_pct",
    "l1_tex_pct",
    "l2_pct",
    "mem_gbs",
    "mem_busy_pct",
    "l1_hit_pct",
    "l2_hit_pct",
    "registers_per_thread",
    "static_smem_kib",
    "dynamic_smem_kib",
    "waves_per_sm",
    "theoretical_occ_pct",
    "achieved_occ_pct",
    "achieved_active_warps_sm",
    "eligible_warps_per_scheduler",
    "no_eligible_pct",
    "issued_warp_per_scheduler",
    "global_excessive_sectors",
    "global_excessive_pct",
    "shared_bank_conflicts",
    "shared_bank_conflict_pct",
    "top_rule",
    "likely_bottleneck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate NCU profiling, details CSV export, and summary extraction."
    )
    parser.add_argument(
        "--bench",
        type=Path,
        default=Path("release/cuda/PMPP/gemm/PMPP_gemm_bench"),
        help="Path to PMPP_gemm_bench. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .ncu-rep, details CSV, and summaries. "
        "Default: <bench-dir>/ncu_profiles",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "profile", "import", "summarize"),
        default="all",
        help="all=profile+import+summarize. import assumes .ncu-rep files exist. "
        "summarize assumes details CSV files exist.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.name for variant in DEFAULT_VARIANTS],
        help="Variant to process. May be repeated. Default: all custom tiled variants.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        type=Path,
        help="Extra details CSV to include in summary mode. The file stem is used as "
        "the variant name unless --csv-label is also supplied.",
    )
    parser.add_argument(
        "--csv-label",
        action="append",
        help="Label for the corresponding --csv argument.",
    )
    parser.add_argument(
        "--ncu",
        default="ncu",
        help="Nsight Compute CLI executable. Default: %(default)s",
    )
    parser.add_argument(
        "--kernel-regex",
        default=None,
        help="Override kernel regex for all variants. Default matches the tiled template.",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=None,
        help="NCU section to collect. May be repeated. Default is a focused section set.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        default="1x",
        help="Google Benchmark --benchmark_min_time value under NCU. Default: %(default)s",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1024,
        help="M dimension used for derived TFLOP/s. Default: %(default)s",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1024,
        help="N dimension used for derived TFLOP/s. Default: %(default)s",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1024,
        help="K dimension used for derived TFLOP/s. Default: %(default)s",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite NCU report files by passing -f to ncu.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one profile/import step fails.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List known variants and exit.",
    )
    return parser.parse_args()


def selected_variants(args: argparse.Namespace) -> list[Variant]:
    if args.mode == "summarize" and args.csv and not args.variant:
        return []
    variants = list(DEFAULT_VARIANTS)
    if args.variant:
        wanted = set(args.variant)
        variants = [variant for variant in variants if variant.name in wanted]
    if args.kernel_regex:
        variants = [
            Variant(variant.name, variant.benchmark, args.kernel_regex)
            for variant in variants
        ]
    return variants


def run_command(
    command: list[str],
    *,
    stdout_path: Path | None = None,
    stop_on_error: bool,
) -> bool:
    print("+ " + " ".join(command))
    try:
        if stdout_path is None:
            subprocess.run(command, check=True)
        else:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w", encoding="utf-8", newline="") as output:
                subprocess.run(command, check=True, stdout=output)
    except subprocess.CalledProcessError as error:
        print(f"warning: command failed with exit code {error.returncode}", file=sys.stderr)
        if stop_on_error:
            raise
        return False
    return True


def profile_variant(
    args: argparse.Namespace, variant: Variant, output_dir: Path
) -> bool:
    report_base = output_dir / f"ncu_sgemm_{variant.name}"
    command = [
        args.ncu,
        "--target-processes",
        "all",
    ]
    if args.force:
        command.append("-f")
    for section in args.section or DEFAULT_SECTIONS:
        command.extend(["--section", section])
    command.extend(
        [
            "--kernel-name",
            f"regex:{variant.kernel_regex}",
            "-o",
            str(report_base),
            str(args.bench),
            f"--benchmark_filter={variant.benchmark}",
            f"--benchmark_min_time={args.benchmark_min_time}",
        ]
    )
    return run_command(command, stop_on_error=args.stop_on_error)


def import_variant(args: argparse.Namespace, variant: Variant, output_dir: Path) -> bool:
    report_path = output_dir / f"ncu_sgemm_{variant.name}.ncu-rep"
    csv_path = output_dir / f"ncu_sgemm_{variant.name}_details.csv"
    if not report_path.exists():
        print(f"warning: missing report {report_path}", file=sys.stderr)
        return False
    command = [args.ncu, "--import", str(report_path), "--page", "details", "--csv"]
    return run_command(
        command, stdout_path=csv_path, stop_on_error=args.stop_on_error
    )


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def normalize_kib(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    unit = (unit or "").lower()
    if unit.startswith("byte"):
        return value / 1024.0
    if unit.startswith("kbyte"):
        return value
    if unit.startswith("mbyte"):
        return value * 1024.0
    return value


def extract_rule_numbers(summary: dict[str, object], row: dict[str, str]) -> None:
    rule = row.get("Rule Name", "")
    description = row.get("Rule Description", "")
    speedup = numeric(row.get("Estimated Speedup"))

    if rule and speedup is not None:
        current = numeric(str(summary.get("_top_rule_speedup", "")))
        if current is None or speedup > current:
            summary["_top_rule_speedup"] = speedup
            summary["top_rule"] = rule

    if rule == "UncoalescedGlobalAccess":
        match = re.search(
            r"total of ([0-9.,]+) excessive sectors \(([0-9.]+)%",
            description,
        )
        if match:
            summary["global_excessive_sectors"] = numeric(match.group(1))
            summary["global_excessive_pct"] = numeric(match.group(2))

    if rule == "SharedMemoryConflicts":
        match = re.search(
            r"results in ([0-9.,]+) bank conflicts,.*represent ([0-9.]+)%",
            description,
        )
        if match:
            summary["shared_bank_conflicts"] = numeric(match.group(1))
            summary["shared_bank_conflict_pct"] = numeric(match.group(2))

    if rule == "CPIStall":
        match = re.search(
            r"spends ([0-9.]+) cycles being stalled.*?due to ([^.]+)\.",
            description,
        )
        if match:
            summary["_stall_cycles"] = numeric(match.group(1))
            summary["_stall_reason"] = match.group(2)


def metric_lookup(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        section = row.get("Section Name", "")
        metric = row.get("Metric Name", "")
        if section and metric:
            lookup[(section, metric)] = row
    return lookup


def read_metric(
    lookup: dict[tuple[str, str], dict[str, str]],
    section: str,
    metric: str,
) -> tuple[float | None, str | None]:
    row = lookup.get((section, metric))
    if row is None:
        return None, None
    return numeric(row.get("Metric Value")), row.get("Metric Unit")


def set_metric(
    summary: dict[str, object],
    key: str,
    lookup: dict[tuple[str, str], dict[str, str]],
    section: str,
    metric: str,
    *,
    kib: bool = False,
) -> None:
    value, unit = read_metric(lookup, section, metric)
    if kib:
        value = normalize_kib(value, unit)
    summary[key] = value


def diagnose(summary: dict[str, object]) -> str:
    compute = numeric(str(summary.get("compute_sm_pct", "")))
    memory = numeric(str(summary.get("memory_sol_pct", "")))
    dram = numeric(str(summary.get("dram_pct", "")))
    achieved_occ = numeric(str(summary.get("achieved_occ_pct", "")))
    waves = numeric(str(summary.get("waves_per_sm", "")))
    bank_pct = numeric(str(summary.get("shared_bank_conflict_pct", "")))
    excessive_pct = numeric(str(summary.get("global_excessive_pct", "")))
    no_eligible = numeric(str(summary.get("no_eligible_pct", "")))

    reasons: list[str] = []
    if waves is not None and waves < 1.0:
        reasons.append("whole-GPU underfill")
    if achieved_occ is not None and achieved_occ < 25.0:
        reasons.append("low achieved occupancy")
    if bank_pct is not None and bank_pct >= 25.0:
        reasons.append("shared-memory bank conflicts")
    if excessive_pct is not None and excessive_pct >= 25.0:
        reasons.append("uncoalesced global access")
    if no_eligible is not None and no_eligible >= 40.0:
        reasons.append("scheduler stalls")
    if compute is not None and memory is not None:
        if memory > compute * 1.2:
            if dram is not None and dram < 20.0:
                reasons.append("L1/shared-memory pressure")
            else:
                reasons.append("memory throughput pressure")
        elif compute > memory * 1.2:
            reasons.append("compute/instruction pressure")
        else:
            reasons.append("balanced SM/memory pressure")

    if not reasons:
        return "insufficient counters"
    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return "; ".join(deduped[:4])


def summarize_csv(label: str, csv_path: Path, dims: tuple[int, int, int]) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    lookup = metric_lookup(rows)
    summary: dict[str, object] = {"variant": label}

    set_metric(
        summary,
        "duration_us",
        lookup,
        "GPU Speed Of Light Throughput",
        "Duration",
    )
    set_metric(
        summary,
        "compute_sm_pct",
        lookup,
        "GPU Speed Of Light Throughput",
        "Compute (SM) Throughput",
    )
    set_metric(
        summary,
        "memory_sol_pct",
        lookup,
        "GPU Speed Of Light Throughput",
        "Memory Throughput",
    )
    set_metric(summary, "dram_pct", lookup, "GPU Speed Of Light Throughput", "DRAM Throughput")
    set_metric(
        summary,
        "l1_tex_pct",
        lookup,
        "GPU Speed Of Light Throughput",
        "L1/TEX Cache Throughput",
    )
    set_metric(summary, "l2_pct", lookup, "GPU Speed Of Light Throughput", "L2 Cache Throughput")
    set_metric(summary, "mem_gbs", lookup, "Memory Workload Analysis", "Memory Throughput")
    set_metric(summary, "mem_busy_pct", lookup, "Memory Workload Analysis", "Mem Busy")
    set_metric(summary, "l1_hit_pct", lookup, "Memory Workload Analysis", "L1/TEX Hit Rate")
    set_metric(summary, "l2_hit_pct", lookup, "Memory Workload Analysis", "L2 Hit Rate")
    set_metric(summary, "registers_per_thread", lookup, "Launch Statistics", "Registers Per Thread")
    set_metric(
        summary,
        "static_smem_kib",
        lookup,
        "Launch Statistics",
        "Static Shared Memory Per Block",
        kib=True,
    )
    set_metric(
        summary,
        "dynamic_smem_kib",
        lookup,
        "Launch Statistics",
        "Dynamic Shared Memory Per Block",
        kib=True,
    )
    set_metric(summary, "waves_per_sm", lookup, "Launch Statistics", "Waves Per SM")
    set_metric(summary, "theoretical_occ_pct", lookup, "Occupancy", "Theoretical Occupancy")
    set_metric(summary, "achieved_occ_pct", lookup, "Occupancy", "Achieved Occupancy")
    set_metric(
        summary,
        "achieved_active_warps_sm",
        lookup,
        "Occupancy",
        "Achieved Active Warps Per SM",
    )
    set_metric(
        summary,
        "eligible_warps_per_scheduler",
        lookup,
        "Scheduler Statistics",
        "Eligible Warps Per Scheduler",
    )
    set_metric(summary, "no_eligible_pct", lookup, "Scheduler Statistics", "No Eligible")
    set_metric(
        summary,
        "issued_warp_per_scheduler",
        lookup,
        "Scheduler Statistics",
        "Issued Warp Per Scheduler",
    )

    duration = numeric(str(summary.get("duration_us", "")))
    if duration and duration > 0:
        m, n, k = dims
        summary["derived_tflops"] = (2.0 * m * n * k) / (duration * 1.0e6)
    else:
        summary["derived_tflops"] = None

    for row in rows:
        extract_rule_numbers(summary, row)

    summary["likely_bottleneck"] = diagnose(summary)
    for key in list(summary):
        if key.startswith("_"):
            del summary[key]
    return summary


def write_summary_csv(summaries: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({column: fmt(summary.get(column)) for column in SUMMARY_COLUMNS})


def markdown_table(rows: list[dict[str, object]], columns: tuple[str, ...]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_summary_md(
    summaries: list[dict[str, object]],
    output_path: Path,
    dims: tuple[int, int, int],
) -> None:
    main_columns = (
        "variant",
        "duration_us",
        "derived_tflops",
        "compute_sm_pct",
        "memory_sol_pct",
        "dram_pct",
        "achieved_occ_pct",
        "registers_per_thread",
        "dynamic_smem_kib",
        "likely_bottleneck",
    )
    memory_columns = (
        "variant",
        "l1_hit_pct",
        "l2_hit_pct",
        "global_excessive_pct",
        "shared_bank_conflict_pct",
        "eligible_warps_per_scheduler",
        "no_eligible_pct",
        "top_rule",
    )

    text = f"""# NCU SGEMM Summary

Matrix size used for derived throughput: M={dims[0]}, N={dims[1]}, K={dims[2]}.

NCU may replay kernels and collect counters in multiple passes, so use
`derived_tflops` for profile-context comparison only. Use Google Benchmark for
clean timing.

## Main Counters

{markdown_table(summaries, main_columns)}

## Memory And Scheduler Signals

{markdown_table(summaries, memory_columns)}

## Reading Guide

- `duration_us`: NCU-reported kernel duration.
- `derived_tflops`: `2*M*N*K / duration`; useful for relative profile comparison.
- `compute_sm_pct` vs `memory_sol_pct`: first split between compute pressure and memory hierarchy pressure.
- `dram_pct`: if this is low while memory SOL is high, the pressure is usually closer to L1/TEX, shared memory, or replay than HBM bandwidth.
- `registers_per_thread`, `dynamic_smem_kib`, and `achieved_occ_pct`: resource pressure and occupancy context.
- `global_excessive_pct`: NCU's uncoalesced global access signal parsed from Source Counters rules.
- `shared_bank_conflict_pct`: shared-memory bank conflict signal parsed from Memory Workload Analysis rules.
- `eligible_warps_per_scheduler` and `no_eligible_pct`: whether schedulers had ready work.
"""
    output_path.write_text(text, encoding="utf-8")


def collect_csv_inputs(
    args: argparse.Namespace,
    variants: list[Variant],
    output_dir: Path,
) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for variant in variants:
        csv_path = output_dir / f"ncu_sgemm_{variant.name}_details.csv"
        if csv_path.exists():
            inputs.append((variant.name, csv_path))
        elif args.mode in ("summarize", "all"):
            print(f"warning: missing CSV {csv_path}", file=sys.stderr)

    extra_csvs = args.csv or []
    labels = args.csv_label or []
    if labels and len(labels) != len(extra_csvs):
        raise SystemExit("--csv-label must be supplied the same number of times as --csv")
    for index, csv_path in enumerate(extra_csvs):
        label = labels[index] if labels else csv_path.stem
        inputs.append((label, csv_path))
    return inputs


def main() -> int:
    args = parse_args()

    if args.list_variants:
        for variant in DEFAULT_VARIANTS:
            print(f"{variant.name}: {variant.benchmark}")
        return 0

    variants = selected_variants(args)
    bench_dir = args.bench.parent
    output_dir = args.output_dir or (bench_dir / "ncu_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "profile", "import") and shutil.which(args.ncu) is None:
        raise SystemExit(f"could not find Nsight Compute CLI: {args.ncu}")

    if args.mode in ("all", "profile") and not args.bench.exists():
        raise SystemExit(f"benchmark executable does not exist: {args.bench}")

    if args.mode in ("all", "profile"):
        for variant in variants:
            profile_variant(args, variant, output_dir)

    if args.mode in ("all", "import"):
        for variant in variants:
            import_variant(args, variant, output_dir)

    if args.mode in ("all", "summarize"):
        csv_inputs = collect_csv_inputs(args, variants, output_dir)
        summaries = [
            summarize_csv(label, csv_path, (args.m, args.n, args.k))
            for label, csv_path in csv_inputs
        ]
        if not summaries:
            print("warning: no CSV inputs found; no summary written", file=sys.stderr)
            return 1
        write_summary_csv(summaries, output_dir / "summary.csv")
        write_summary_md(summaries, output_dir / "summary.md", (args.m, args.n, args.k))
        print(f"wrote {output_dir / 'summary.csv'}")
        print(f"wrote {output_dir / 'summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
