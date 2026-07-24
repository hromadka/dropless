#!/usr/bin/env python3
"""
batch.py
========
Sweep total weapons M (= D + I) and target count n, then plot surjective
assignment solution counts from surjective_assignments.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from surjective_assignments import surjective_count


def split_weapons(M: int, d_ref: int, i_ref: int) -> tuple[int, int]:
    """Split M weapons into distinguishable and indistinguishable counts."""
    d = M * d_ref // (d_ref + i_ref)
    return d, M - d


def build_types(d: int, i: int) -> list[tuple[int, bool]]:
    types: list[tuple[int, bool]] = []
    if d > 0:
        types.append((d, True))
    if i > 0:
        types.append((i, False))
    return types


def compute_grid(
    d_ref: int,
    i_ref: int,
    m_max: int = 10,
    n_max: int = 10,
    lo: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid M, N and solution counts Z."""
    m_vals = np.arange(lo, m_max + 1)
    n_vals = np.arange(lo, n_max + 1)
    M, N = np.meshgrid(m_vals, n_vals, indexing="xy")
    Z = np.zeros_like(M, dtype=np.float64)

    for idx, m in np.ndenumerate(M):
        n = N[idx]
        d, i = split_weapons(int(m), d_ref, i_ref)
        types = build_types(d, i)
        if types:
            Z[idx] = surjective_count(types, int(n))

    return M, N, Z


def _mix_label(d_ref: int, i_ref: int) -> str:
    return f"D:I = {d_ref}:{i_ref}"


def _output_suffix(m_max: int, n_max: int, d_ref: int, i_ref: int) -> str:
    return f"_M{m_max}n{n_max}_ratio{d_ref}-{i_ref}"


def _set_m_axis_descending(ax, M: np.ndarray) -> None:
    ax.set_xlim(int(M.max()), int(M.min()))


def _z_for_log_plot(Z: np.ndarray) -> np.ndarray:
    """Return Z with non-positive values masked for log-scale plotting."""
    z_plot = np.array(Z, dtype=np.float64, copy=True)
    z_plot[z_plot <= 0] = np.nan
    return z_plot


def _log_norm(Z: np.ndarray, z_max: float | None = None) -> LogNorm | None:
    positive = Z[Z > 0]
    if positive.size == 0 and z_max is None:
        return None
    vmin = float(positive.min()) if positive.size > 0 else 1.0
    vmax = float(z_max) if z_max is not None else float(positive.max())
    return LogNorm(vmin=vmin, vmax=vmax)


def _configure_log_z_axis(
    ax,
    Z: np.ndarray,
    z_max: float | None = None,
) -> None:
    positive = Z[Z > 0]
    z_min = float(positive.min()) if positive.size > 0 else 1.0
    z_top = float(z_max) if z_max is not None else (
        float(positive.max()) if positive.size > 0 else z_min
    )
    ax.set_zlim(z_min, z_top)
    ax.set_zscale("log")
    ax.set_zlabel("Solution count (log scale)")


def plot_surface(
    M: np.ndarray,
    N: np.ndarray,
    Z: np.ndarray,
    d_ref: int,
    i_ref: int,
    outfile: Path,
    z_max: float | None = None,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    z_plot = _z_for_log_plot(Z)
    norm = _log_norm(Z, z_max)
    surf = ax.plot_surface(
        M,
        N,
        z_plot,
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    ax.set_xlabel("m (total weapon types, D + I)")
    ax.set_ylabel("n (targets)")
    ax.set_title(f"Surjective assignment counts ({_mix_label(d_ref, i_ref)})")
    _set_m_axis_descending(ax, M)
    _configure_log_z_axis(ax, Z, z_max)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Solution count (log scale)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_scatter(
    M: np.ndarray,
    N: np.ndarray,
    Z: np.ndarray,
    d_ref: int,
    i_ref: int,
    outfile: Path,
    z_max: float | None = None,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    mask = Z.ravel() > 0
    norm = _log_norm(Z, z_max)
    scatter = ax.scatter(
        M.ravel()[mask],
        N.ravel()[mask],
        Z.ravel()[mask],
        c=Z.ravel()[mask],
        cmap="viridis",
        norm=norm,
        s=50,
        depthshade=True,
    )
    ax.set_xlabel("M (total weapons, D + I)")
    ax.set_ylabel("n (targets)")
    ax.set_title(f"Surjective assignment counts ({_mix_label(d_ref, i_ref)})")
    _set_m_axis_descending(ax, M)
    _configure_log_z_axis(ax, Z, z_max)
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Solution count (log scale)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_xyz(
    M: np.ndarray,
    N: np.ndarray,
    Z: np.ndarray,
    d_ref: int,
    i_ref: int,
    outfile: Path,
) -> None:
    """Write M, D, I, n, and solution count columns to a text file."""
    with outfile.open("w", encoding="utf-8") as f:
        f.write(f"# d_ref={d_ref}\ti_ref={i_ref}\n")
        f.write("M\tD\tI\tn\tsolution_count\n")
        for m, n, z in zip(M.ravel(), N.ravel(), Z.ravel(), strict=True):
            d, i = split_weapons(int(m), d_ref, i_ref)
            f.write(f"{int(m)}\t{d}\t{i}\t{int(n)}\t{int(z)}\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep total weapons M (= D + I) and target count n, then plot "
            "surjective assignment solution counts."
        ),
    )
    parser.add_argument(
        "-D",
        "--distinguishable",
        type=int,
        default=1,
        metavar="D",
        help="reference distinguishable weapon count for the D:I mix (default: 1)",
    )
    parser.add_argument(
        "-I",
        "--indistinguishable",
        type=int,
        default=1,
        metavar="I",
        help="reference indistinguishable weapon count for the D:I mix (default: 1)",
    )
    parser.add_argument(
        "-M",
        "--max-weapons",
        type=int,
        default=10,
        metavar="M",
        help="maximum total weapons M to sweep (default: 10)",
    )
    parser.add_argument(
        "-n",
        "--max-targets",
        type=int,
        default=10,
        metavar="N",
        help="maximum target count n to sweep (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for output PNG and text files (default: current directory)",
    )
    parser.add_argument(
        "--z-max-10m",
        action="store_true",
        help="cap the z-axis upper limit at 10^M, where M is -M/--max-weapons",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.distinguishable < 0:
        raise SystemExit("error: -D/--distinguishable must be >= 0")
    if args.indistinguishable < 0:
        raise SystemExit("error: -I/--indistinguishable must be >= 0")
    if args.distinguishable + args.indistinguishable < 1:
        raise SystemExit("error: at least one of -D or -I must be > 0")
    if args.max_weapons < 1:
        raise SystemExit("error: -M/--max-weapons must be >= 1")
    if args.max_targets < 1:
        raise SystemExit("error: -n/--max-targets must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Computing surjective counts for M in [1, {args.max_weapons}], "
        f"n in [1, {args.max_targets}] with mix {_mix_label(args.distinguishable, args.indistinguishable)}..."
    )
    M, N, Z = compute_grid(
        args.distinguishable,
        args.indistinguishable,
        args.max_weapons,
        args.max_targets,
    )

    suffix = _output_suffix(
        args.max_weapons,
        args.max_targets,
        args.distinguishable,
        args.indistinguishable,
    )
    surface_path = args.output_dir / f"surjective_surface{suffix}.png"
    scatter_path = args.output_dir / f"surjective_scatter{suffix}.png"
    data_path = args.output_dir / f"surjective_data{suffix}.txt"

    save_xyz(M, N, Z, args.distinguishable, args.indistinguishable, data_path)

    z_max = float(10 ** args.max_weapons) if args.z_max_10m else None
    plot_surface(
        M, N, Z, args.distinguishable, args.indistinguishable, surface_path, z_max
    )
    plot_scatter(
        M, N, Z, args.distinguishable, args.indistinguishable, scatter_path, z_max
    )

    print(f"Data saved to {data_path}")
    print(f"Surface plot saved to {surface_path}")
    print(f"Scatter plot saved to {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
