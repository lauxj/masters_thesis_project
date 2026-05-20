"""
Create evaluation plots from saved EWFS experiment runs.

Inputs:
- data/paperdata by default, for reproducing the thesis figures
- data/data_noiseless_simulation, data/data_fake_hardware, and
  data/data_real_hardware for newly generated runs
- saved lf_violations/lf_violations.json files
- optional accuracy-test result files

Outputs:
- PNG/PDF plots under results/plots/plots_agent_evaluation/<timestamp>
- JSON sidecar metadata for generated plots

By default, the script uses the frozen paper-data runs. Use
--data-source latest-runs to evaluate newly generated runs instead. Specific
runs can be selected with --noiseless-run, --fake-run, and --real-run.

File map:
- configuration constants
- small plotting and JSON helpers
- run selection / data loading
- metric extraction and aggregation
- plot builders
- printed summaries and main()
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

# directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_REAL = PROJECT_ROOT / "data" / "data_real_hardware"
DATA_DIR_NOISELESS = PROJECT_ROOT / "data" / "data_noiseless_simulation"
DATA_DIR_FAKE = PROJECT_ROOT / "data" / "data_fake_hardware"
PAPERDATA_ROOT = PROJECT_ROOT / "data" / "paperdata"
PAPERDATA_DIR_NOISELESS = PAPERDATA_ROOT / "noiseless_simulation"
PAPERDATA_DIR_FAKE = PAPERDATA_ROOT / "fake_hardware"
PAPERDATA_DIR_REAL = PAPERDATA_ROOT / "real_hardware"
PLOTS_ROOT = PROJECT_ROOT / "results" / "plots" / "plots_agent_evaluation"

# -----------------------------------------------------------------------------
# Configuration

# Thesis plot typography. Keep all plot families on one scale so the evaluation
# section reads coherently when figures are placed side by side.
PLOT_FONT_SIZE_DELTA = 0
THESIS_BASE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 12
LF_LEGEND_FONT_SIZE = 13
LF_PANEL_LABEL_FONT_SIZE = 14
LF_VALUE_LABEL_FONT_SIZE = 11
LF_LOWER_TICK_FONT_SIZE = 16
LF_SUMMARY_LABEL_FONT_SIZE = 15
LF_SUMMARY_VALUE_LABEL_FONT_SIZE = 12
LF_SUMMARY_LEGEND_FONT_SIZE = 14
LF_SUMMARY_AGENT_LABEL_X = -0.085
SINGLE_PANEL_FIGSIZE = (9.2, 5.6)
LF_SINGLE_BACKEND_FIGSIZE = (10.0, 6.1)
LF_COMPARISON_FIGSIZE = (10.0, 6.4)
LF_SUMMARY_FIGURE_WIDTH = 10.0
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": THESIS_BASE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})


def plot_fontsize(points: int) -> int:
    return points + PLOT_FONT_SIZE_DELTA

# Choose which saved data to evaluate.
# "paperdata" uses the frozen thesis runs in data/paperdata.
# "latest-runs" uses the newest runs in data/data_noiseless_simulation,
# data/data_fake_hardware, and data/data_real_hardware.
EVALUATION_DATA_SOURCE = "paperdata"
#EVALUATION_DATA_SOURCE = "latest-runs"

# Number of runs to aggregate per backend. This takes the n-latest runs, so make
# sure those runs are from one backend and close together in time.
EVALUATION_LAST_N = {
    "Noiseless": 10,
    "Fake hardware": 10,
    "Real hardware": 10,
}
# If run folders are inserted here, they are used instead of EVALUATION_LAST_N.
EVALUATION_RUN_PATHS = {
    "Noiseless": [],
    "Fake hardware": [],
    "Real hardware": [],
}


#-----------------------------------------------------------------------------
# Constants and small helpers for plots

IDEAL_COLOR = "#222222"
THEORY_LINE_COLOR = "#C92A2A"
ACCURACY_THEORY_LINE_COLOR = "#FF0000"
PAYOFF_BY_WALLET_STATE = {
    "00": -3 / 4,
    "01": -1 / 4,
    "10": 1 / 4,
    "11": 3 / 4,
}
THEORY_C1_PROBABILITIES = {
    "0": 0.5,
    "1": 0.5,
}

STAKE_BY_WALLET_STATE = {
    "00": 3 / 4,
    "01": 1 / 4,
    "10": 3 / 4,
    "11": 1 / 4,
}
# Betting circuit count strings are read as c[7]...c[0].
# M1 is measured into c[2], which is index 5 from the left.
M1_BIT_INDEX_FROM_LEFT = 5

# Guessing circuit count strings are read as c[6]...c[0].
# M2 is measured into c[5] -> index 1 from the left.
# G is measured into c[6] -> index 0 from the left.
GUESSING_M2_INDEX_FROM_LEFT = 1
GUESSING_G_INDEX_FROM_LEFT = 0

# Reflex circuit count strings are read as c[5]...c[0].
# R is measured into c[5] -> index 0 from the left.
# M is measured into c[2] -> index 3 from the left.
REFLEX_SA_INDEX_FROM_LEFT = 2
REFLEX_R_INDEX_FROM_LEFT = 0
REFLEX_M_INDEX_FROM_LEFT = 3

LF_AGENT_NAMES = ["Betting Agent", "Guessing Agent", "Reflex Agent", "Always 3/4 Agent"]
STANDARD_AGENT_NAMES = LF_AGENT_NAMES
MEMORY_PLOT_AGENT_NAMES = ["Reflex Agent", "Guessing Agent", "Always 3/4 Agent", "Betting Agent"]
HARDWARE_LF_SUMMARY_AGENT_NAMES = ["Reflex Agent", "Guessing Agent", "Betting Agent", "Always 3/4 Agent"]
AGENT_DISPLAY_LABELS = {
    "Always 3/4 Agent": "Always-3/4 Agent",
}
LF_TERM_SPECS = [
    ("E11", -1.0, r"$-\langle A_1 B_1 \rangle$"),
    ("E12", 1.0, r"$\langle A_1 B_2 \rangle$"),
    ("E21", -1.0, r"$-\langle A_2 B_1 \rangle$"),
    ("E22", -1.0, r"$-\langle A_2 B_2 \rangle$"),
]
LF_TERM_COLORS = ["#FFF3BF", "#5C7CFA", "#D0EBFF", "#9C36B5"]
LF_CORRELATOR_LABELS = [
    r"$\langle A_1 B_1 \rangle$",
    r"$\langle A_1 B_2 \rangle$",
    r"$\langle A_2 B_1 \rangle$",
    r"$\langle A_2 B_2 \rangle$",
]
LF_ANALYTIC_CORRELATORS = {
    "E11": -1.0 / np.sqrt(2.0),
    "E12": 1.0 / np.sqrt(2.0),
    "E21": -1.0 / np.sqrt(2.0),
    "E22": -1.0 / np.sqrt(2.0),
}
BACKEND_COLORS = {
    "Noiseless": "#1F77B4",
    "Fake hardware": "#FF7F0E",
    "Real hardware": "#2CA02C",
}
MEMORY_BACKEND_COLORS = {
    "Noiseless": "#14855F",
    "Fake hardware": "#2B83BA",
    "Real hardware": "#8E37AD",
}
ACCURACY_BACKEND_COLORS = {
    "Noiseless": "#14855F",
    "Fake hardware": "#2B83BA",
    "Real hardware": "#8E37AD",
}
BACKEND_LABELS = ["Noiseless", "Fake hardware", "Real hardware"]
BACKEND_DISPLAY_LABELS = {
    "Noiseless": "Ideal simulation",
}
BACKEND_AXIS_LABELS = {
    "Noiseless": "Ideal\nsimulation",
}
THEORY_COMPARISON_COLORS = {
    "Born-rule": "#9467BD",
    "Random": "#7F7F7F",
    "Opposite": "#8C564B",
    "Always 1/4": "#17BECF",
    "Always 3/4": "#BCBD22",
}
ACCURACY_TEST_RESULT_FILENAMES = {
    "Noiseless": "accuracy_test_noiseless_simulation.json",
    "Fake hardware": "accuracy_test_fake_hardware_noise_sim.json",
    "Real hardware": "accuracy_test_real_hardware_run.json",
}
ACCURACY_TEST_SUFFIX_INIT0 = "_accuracy_test_init0"
ACCURACY_TEST_SUFFIX_INIT1 = "_accuracy_test_init1"
PAYOFF_COLORS = [
    BACKEND_COLORS["Noiseless"],
    BACKEND_COLORS["Fake hardware"],
    BACKEND_COLORS["Real hardware"],
    THEORY_COMPARISON_COLORS["Random"],
    THEORY_COMPARISON_COLORS["Opposite"],
    THEORY_COMPARISON_COLORS["Always 1/4"],
    THEORY_COMPARISON_COLORS["Always 3/4"],
]
# Toggle these to restore the cleaner LF figures without the 4 epsilon guides.
SHOW_FOUR_EPSILON_IN_BACKEND_LF_PLOTS = False
SHOW_FOUR_EPSILON_IN_HARDWARE_LF_COMPARISON_PLOTS = False
SHOW_FOUR_EPSILON_IN_HARDWARE_LF_SUMMARY_PLOT = True
ACCURACY_METRIC_SPECS = {
    "guessing_accuracy": {
        "value_key": "guessing_accuracy",
        "error_key": "guessing_accuracy_stderr",
        "shots_key": "guessing_accuracy_shots",
        "title": "Guessing Agent Accuracy",
        "ylabel": r"$P(G=M_2)$",
        "ideal_value": 0.75,
        "ideal_label": "Ideal accuracy = 0.75",
        "show_legend": True,
        "y_max": 0.9,
        "filename": "guessing_agent_accuracy_comparison.png",
        "summary_title": "\nGuessing agent accuracy:",
    },
    "reflex_accuracy": {
        "value_key": "reflex_accuracy",
        "error_key": "reflex_accuracy_stderr",
        "shots_key": "reflex_accuracy_shots",
        "title": "Reflex Agent Accuracy",
        "ylabel": r"$P(R=M)$",
        "ideal_value": 1.0,
        "ideal_label": "Ideal accuracy = 1.0",
        "show_legend": True,
        "y_max": 1.2,
        "filename": "reflex_agent_accuracy_comparison.png",
        "summary_title": "\nReflex agent accuracy:",
    },
    "reflex_sa_m_accuracy": {
        "value_key": "reflex_sa_m_accuracy",
        "error_key": "reflex_sa_m_accuracy_stderr",
        "shots_key": "reflex_sa_m_accuracy_shots",
        "title": r"Reflex Agent: $S_A$ and $M$ Agreement",
        "ylabel": r"$P(S_A=M)$",
        "ideal_value": 1.0,
        "ideal_label": r"Ideal agreement = 1.0",
        "show_legend": True,
        "y_max": 1.2,
        "filename": "reflex_agent_sa_m_agreement_accuracy.png",
        "summary_title": "\nReflex agent S_a/M agreement accuracy:",
    },
    "always_large_accuracy": {
        "value_key": "always_large_accuracy",
        "error_key": "always_large_accuracy_stderr",
        "shots_key": "always_large_accuracy_shots",
        "title": "Always-3/4 Agent Accuracy",
        "ylabel": r"$P(\mathrm{bet}=3/4)$",
        "ideal_value": 1.0,
        "ideal_label": "Ideal accuracy = 1.0",
        "show_legend": True,
        "y_max": 1.2,
        "filename": "always_3_4_agent_accuracy_comparison.png",
        "summary_title": "\nAlways-3/4 agent accuracy:",
    },
}


# -----------------------------------------------------------------------------
# Small plotting helpers

def apply_thesis_axis_typography(ax):
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)


def set_thesis_xticklabels(ax, labels, *, rotation=0, ha="center", fontsize: int = TICK_LABEL_FONT_SIZE):
    ax.set_xticklabels(
        labels,
        rotation=rotation,
        ha=ha,
        fontsize=fontsize,
    )


def set_thesis_yticklabels(ax, labels, *, fontsize: int = TICK_LABEL_FONT_SIZE):
    ax.set_yticklabels(labels, fontsize=fontsize)


def style_bar_axes(ax, title: str, ylabel: str):
    ax.set_ylabel(ylabel)
    apply_thesis_axis_typography(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def style_accuracy_axes(ax, ylabel: str):
    ax.set_ylabel(ylabel)
    apply_thesis_axis_typography(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)


def style_memory_axes(ax, ylabel: str):
    ax.set_ylabel(ylabel)
    apply_thesis_axis_typography(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)


def set_probability_axis_ticks(ax, y_max: float):
    if y_max <= 1.001:
        ticks = np.linspace(0.0, y_max, 6)
    else:
        ticks = np.linspace(0.0, 1.0, 6)
    ax.set_yticks(ticks)
    set_thesis_yticklabels(ax, [f"{tick:.1f}" for tick in ticks])


def place_legend_above_axes(fig, ax, *, ncol: int = 1, fontsize: int = 10, handles=None):
    legend_kwargs = {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 1.02),
        "fontsize": plot_fontsize(fontsize),
        "frameon": True,
        "ncol": ncol,
        "borderaxespad": 0.0,
    }
    if handles is not None:
        legend_kwargs["handles"] = handles
    ax.legend(**legend_kwargs)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))


def clean_bitstring(bitstring: str) -> str:
    return "".join(ch for ch in str(bitstring) if ch in {"0", "1"})


def pm(bit: str) -> int:
    return 1 if bit == "0" else -1


def sample_sigma(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def sem_from_values(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(sample_sigma(arr) / np.sqrt(arr.size))


def summarize_measurement(values, single_run_stderr: Optional[float] = None):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarise an empty measurement list.")

    sigma = sample_sigma(arr)
    sem = sem_from_values(arr)
    stderr = float(single_run_stderr or 0.0) if arr.size == 1 else sem
    return {
        "value": float(np.mean(arr)),
        "stderr": stderr,
        "sigma": sigma,
        "sem": sem,
        "sample_count": int(arr.size),
    }


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def sidecar_metadata_path(plot_path: Path) -> Path:
    return plot_path.with_suffix(".json")


def save_plot_metadata(plot_path: Path, metadata: dict) -> Path:
    metadata_path = sidecar_metadata_path(plot_path)
    save_json(metadata_path, metadata)
    return metadata_path


def pdf_plot_path(plot_path: Path) -> Path:
    return plot_path.with_suffix(".pdf")


def save_plot(fig, plot_path: Path, *, dpi: int = 300, bbox_inches: str = "tight") -> Path:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(pdf_plot_path(plot_path), dpi=dpi, bbox_inches=bbox_inches)
    return plot_path


# -----------------------------------------------------------------------------
# Run selection and file loading

def candidate_run_dirs(data_dir: Path, result_filename: str):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    runs = [
        run_dir for run_dir in data_dir.iterdir()
        if run_dir.is_dir() and (run_dir / result_filename).exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No run folders with {result_filename} found in {data_dir.resolve()}"
        )

    return runs


def result_file_has_required_agents(result_path: Path, required_agent_names) -> bool:
    if not required_agent_names:
        return True
    try:
        data = load_json(result_path)
    except (OSError, json.JSONDecodeError):
        return False
    agents = data.get("agents", {})
    return all(agent_name in agents for agent_name in required_agent_names)


def find_latest_run(data_dir: Path, result_filename: str, required_agent_names=None) -> Path:
    runs = candidate_run_dirs(data_dir, result_filename)
    if required_agent_names:
        runs = [
            run_dir for run_dir in runs
            if result_file_has_required_agents(run_dir / result_filename, required_agent_names)
        ]
        if not runs:
            raise FileNotFoundError(
                f"No run folders with {result_filename} and required agents found in {data_dir.resolve()}"
            )
    return max(runs, key=lambda run_dir: run_dir.stat().st_mtime)


def latest_n_runs(data_dir: Path, result_filename: str, count: int, required_agent_names=None):
    if count <= 0:
        raise ValueError("last_n must be a positive integer.")

    runs = candidate_run_dirs(data_dir, result_filename)
    if required_agent_names:
        runs = [
            run_dir for run_dir in runs
            if result_file_has_required_agents(run_dir / result_filename, required_agent_names)
        ]
        if not runs:
            raise FileNotFoundError(
                f"No run folders with {result_filename} and required agents found in {data_dir.resolve()}"
            )
    runs.sort(key=lambda run_dir: (run_dir.stat().st_mtime, run_dir.name), reverse=True)
    return runs[:count]


def resolve_run_path(path_ref: str, data_dir: Path, result_filename: str) -> Path:
    raw_path = Path(path_ref).expanduser()
    candidate_paths = [raw_path.resolve()] if raw_path.is_absolute() else [
        (PROJECT_ROOT / raw_path).resolve(),
        (data_dir / raw_path).resolve(),
    ]

    run_dir = next((path for path in candidate_paths if path.is_dir()), None)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found: {candidate_paths[0]}")
    if not (run_dir / result_filename).exists():
        raise FileNotFoundError(f"Expected result file not found: {(run_dir / result_filename).resolve()}")
    return run_dir


def resolve_manual_run_dir(data_dir: Path, result_filename: str, run_name: str) -> Path:
    run_dir = (data_dir / run_name).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not (run_dir / result_filename).exists():
        raise FileNotFoundError(f"Expected result file not found: {(run_dir / result_filename).resolve()}")
    return run_dir


def resolve_run_dirs(
    label: str,
    data_dir: Path,
    result_filename: str,
    run_name: Optional[str] = None,
    last_n: Optional[int] = None,
) -> Tuple[list[Path], str]:
    if run_name:
        return [resolve_manual_run_dir(data_dir, result_filename, run_name)], "manual"

    run_paths = EVALUATION_RUN_PATHS[label]
    if run_paths:
        return [resolve_run_path(path_ref, data_dir, result_filename) for path_ref in run_paths], "paths"

    resolved_last_n = int(last_n if last_n is not None else EVALUATION_LAST_N[label])
    if resolved_last_n <= 1:
        return [find_latest_run(data_dir, result_filename, required_agent_names=STANDARD_AGENT_NAMES)], "latest"

    return latest_n_runs(
        data_dir,
        result_filename,
        resolved_last_n,
        required_agent_names=STANDARD_AGENT_NAMES,
    ), f"last_{resolved_last_n}"


def resolve_accuracy_test_run_dirs(
    label: str,
    data_dir: Path,
    main_result_filename: str,
    accuracy_test_result_filename: str,
    run_name: Optional[str] = None,
    last_n: Optional[int] = None,
) -> Tuple[list[Path], str]:
    if run_name:
        run_dir = resolve_manual_run_dir(data_dir, main_result_filename, run_name)
        if not (run_dir / accuracy_test_result_filename).exists():
            raise FileNotFoundError(
                f"Accuracy-test result file not found for selected run: "
                f"{(run_dir / accuracy_test_result_filename).resolve()}"
            )
        return [run_dir], "manual"

    run_paths = EVALUATION_RUN_PATHS[label]
    if run_paths:
        run_dirs = [
            resolve_run_path(path_ref, data_dir, main_result_filename)
            for path_ref in run_paths
        ]
        missing = [
            str((run_dir / accuracy_test_result_filename).resolve())
            for run_dir in run_dirs
            if not (run_dir / accuracy_test_result_filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing accuracy-test result files for the explicitly selected runs:\n"
                + "\n".join(missing)
            )
        return run_dirs, "paths"

    run_dirs, selection_mode = resolve_run_dirs(
        label,
        data_dir,
        main_result_filename,
        run_name=None,
        last_n=last_n,
    )
    if all((run_dir / accuracy_test_result_filename).exists() for run_dir in run_dirs):
        return run_dirs, selection_mode

    resolved_last_n = int(last_n if last_n is not None else EVALUATION_LAST_N[label])
    fallback_mode = "latest_accuracy_test_available" if resolved_last_n <= 1 else f"last_{resolved_last_n}_accuracy_test_available"
    if resolved_last_n <= 1:
        return [find_latest_run(data_dir, accuracy_test_result_filename)], fallback_mode
    return latest_n_runs(data_dir, accuracy_test_result_filename, resolved_last_n), fallback_mode


def resolve_lf_result_paths(run_dirs):
    return [run_dir / "lf_violations" / "lf_violations.json" for run_dir in run_dirs]


def result_value(results, label: str, key: str):
    return next(result[key] for result in results if result["label"] == label)


def result_for_label(results, label: str):
    return next(result for result in results if result["label"] == label)


def backend_values(results, key: str):
    return [result_value(results, label, key) for label in BACKEND_LABELS]


def prettify_backend_name(backend_name: Optional[str]) -> str:
    if not backend_name:
        return "IBM backend"

    if backend_name == "mixed_backends":
        return "Mixed IBM backends"

    backend_name = str(backend_name)
    if backend_name.startswith("ibm_"):
        backend_name = backend_name[4:]

    parts = [part.capitalize() for part in backend_name.split("_") if part]
    if not parts:
        return "IBM backend"

    return "IBM " + " ".join(parts)


def infer_backend_name(label: str, run_dir: Path, data: dict) -> Optional[str]:
    if label == "Noiseless":
        return None

    backend_name = data.get("backend")
    if backend_name:
        return str(backend_name)

    job_info_path = run_dir / "job_info.json"
    if job_info_path.exists():
        job_info = load_json(job_info_path)
        backend_name = job_info.get("backend")
        if backend_name:
            return str(backend_name)

    name_parts = run_dir.name.rsplit("_", 2)
    if len(name_parts) == 3:
        return name_parts[0]

    return None


def summarize_backend_name(backend_names) -> Optional[str]:
    unique_names = []
    for backend_name in backend_names:
        if not backend_name:
            continue
        backend_name = str(backend_name)
        if backend_name not in unique_names:
            unique_names.append(backend_name)

    if not unique_names:
        return None
    if len(unique_names) == 1:
        return unique_names[0]
    return "mixed_backends"


def build_backend_display_label(label: str, backend_name: Optional[str]) -> str:
    if label in BACKEND_DISPLAY_LABELS:
        return BACKEND_DISPLAY_LABELS[label]

    backend_title = prettify_backend_name(backend_name)
    if label == "Fake hardware":
        return f"{backend_title} noise simulation"
    if label == "Real hardware":
        return f"{backend_title} hardware"
    return label


def build_backend_axis_label(label: str, backend_name: Optional[str]) -> str:
    if label in BACKEND_AXIS_LABELS:
        return BACKEND_AXIS_LABELS[label]

    backend_title = prettify_backend_name(backend_name)
    if label == "Fake hardware":
        return f"{backend_title}\nnoise simulation"
    if label == "Real hardware":
        return f"{backend_title}\nhardware"
    return build_backend_display_label(label, backend_name)


def result_display_label(result) -> str:
    return result.get("display_label", build_backend_display_label(result["label"], result.get("backend_name")))


def result_axis_label(result) -> str:
    return result.get("axis_label", build_backend_axis_label(result["label"], result.get("backend_name")))


def agent_display_label(agent_name: str) -> str:
    return AGENT_DISPLAY_LABELS.get(agent_name, agent_name)


def multiline_agent_display_label(agent_name: str) -> str:
    return agent_display_label(agent_name).replace(" Agent", "\nAgent")


def backend_lf_panel_label(result) -> str:
    return result_axis_label(result)


def selected_inputs_metadata(results):
    out = []
    for result in results:
        entry = {
            "label": result["label"],
            "display_label": result.get("display_label"),
            "axis_label": result.get("axis_label"),
            "backend_name": result.get("backend_name"),
            "selection_mode": result.get("selection_mode"),
            "run_count": int(result.get("run_count", 0)),
            "run_names": list(result.get("run_names", [])),
            "run_dirs": [str(path) for path in result.get("run_dirs", [])],
            "source_result_paths": [str(path) for path in result.get("source_result_paths", [])],
            "raw_shots_per_run": [int(value) for value in result.get("raw_shots_per_run", [])],
            "raw_shots_total": int(result.get("raw_shots_total", 0)),
        }
        if "lf_result_paths" in result:
            entry["lf_result_paths"] = [str(path) for path in result.get("lf_result_paths", [])]
        if "available" in result:
            entry["available"] = bool(result.get("available", True))
        if "error" in result:
            entry["error"] = result.get("error")
        out.append(entry)
    return out


def common_plot_metadata(results, *, plot_type: str, title: str, description: Optional[str] = None):
    metadata = {
        "plot_type": plot_type,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "uncertainty_definition": (
            "If multiple runs are selected, displayed errors are the standard error of the mean "
            "(SEM) across run-level values. If a single run is selected, displayed errors use "
            "the single-run standard error stored for that metric."
        ),
        "selected_inputs": selected_inputs_metadata(results),
    }
    if description:
        metadata["description"] = description
    return metadata


def build_accuracy_plot_metadata(
    results,
    *,
    plot_type: str,
    title: str,
    value_key: str,
    error_key: str,
    ideal_value: float,
    ideal_label: str,
    ylabel: str,
):
    metadata = common_plot_metadata(results, plot_type=plot_type, title=title)
    metadata["y_axis"] = ylabel
    metadata["ideal_reference"] = {
        "label": ideal_label,
        "value": float(ideal_value),
    }
    metadata["series"] = [
        {
            "backend_label": result["label"],
            "display_label": result["display_label"],
            "value": float(result[value_key]),
            "error": float(result[error_key]),
        }
        for result in results
    ]
    return metadata


def build_accuracy_plot_metadata_for_metric(results, metric_key: str):
    spec = ACCURACY_METRIC_SPECS[metric_key]
    return build_accuracy_plot_metadata(
        results,
        plot_type=metric_key,
        title=spec["title"],
        value_key=spec["value_key"],
        error_key=spec["error_key"],
        ideal_value=spec["ideal_value"],
        ideal_label=spec["ideal_label"],
        ylabel=spec["ylabel"],
    )


def build_memory_epsilon_plot_metadata(memory_inaccuracy_results):
    summary = build_memory_inaccuracy_summary(memory_inaccuracy_results)
    metadata = common_plot_metadata(
        memory_inaccuracy_results,
        plot_type="combined_memory_initialization_epsilon",
        title=r"$P(c \neq a \mid x=1)$ estimate",
        description=(
            "For each agent, init0 and init1 accuracy-test results are combined as "
            "0.5 * (accuracy_init0 + accuracy_init1). With P(a=0)=P(a=1)=0.5, "
            "the plotted quantity is P(c != a) for the final A = c estimate."
        ),
    )
    metadata["y_axis"] = r"$P(c \neq a \mid x=1)$"
    metadata["categories"] = []
    backend_map = {
        backend["label"]: backend
        for backend in summary["backends"]
    }
    for agent_name in MEMORY_PLOT_AGENT_NAMES:
        category = {"agent_name": agent_name, "series": []}
        for backend_label in BACKEND_LABELS:
            backend = backend_map[backend_label]
            combined = backend.get("combined_agents", {}).get(agent_name)
            if combined is None:
                continue
            category["series"].append(
                {
                    "backend_label": backend_label,
                    "display_label": backend["display_label"],
                    "epsilon": float(combined["epsilon"]),
                    "epsilon_stderr": float(combined["epsilon_stderr"]),
                    "mean_init_accuracy": float(combined["mean_init_accuracy"]),
                    "mean_init_accuracy_stderr": float(combined["mean_init_accuracy_stderr"]),
                }
            )
        metadata["categories"].append(category)
    return metadata


def build_born_rule_plot_metadata(results):
    metadata = common_plot_metadata(
        results,
        plot_type="born_rule_accuracy",
        title="Betting Agent Accuracy",
    )
    metadata["ideal_reference"] = {"label": "Ideal Betting Agent", "value": 1.0}
    metadata["categories"] = [
        {
            "category": "P(bet 1/4 | c1=0)",
            "series": [
                {
                    "backend_label": result["label"],
                    "display_label": result["display_label"],
                    "value": float(result["strategy_probabilities"]["P(bet 1/4 | c1=0)"]),
                    "error": float(result["strategy_probabilities"]["P(bet 1/4 | c1=0) stderr"]),
                }
                for result in results
            ],
        },
        {
            "category": "P(bet 3/4 | c1=1)",
            "series": [
                {
                    "backend_label": result["label"],
                    "display_label": result["display_label"],
                    "value": float(result["strategy_probabilities"]["P(bet 3/4 | c1=1)"]),
                    "error": float(result["strategy_probabilities"]["P(bet 3/4 | c1=1) stderr"]),
                }
                for result in results
            ],
        },
    ]
    return metadata


def build_payoff_comparison_metadata(results):
    metadata = common_plot_metadata(
        results,
        plot_type="payoff_comparison",
        title="Betting Agent vs Always-3/4 Payoff",
    )
    metadata["theory"] = {
        "Betting Agent": float(theory_payoff_for_policy("betting")),
        "Always-3/4 agent": float(theory_payoff_for_policy("always_large")),
    }
    metadata["series"] = [
        {
            "backend_label": result["label"],
            "display_label": result["display_label"],
            "born_rule_payoff": float(result["observed_payoff"]),
            "born_rule_error": float(result["observed_payoff_stderr"]),
            "always_large_payoff": float(result["always_large_observed_payoff"]),
            "always_large_error": float(result["always_large_observed_payoff_stderr"]),
        }
        for result in results
    ]
    return metadata


def build_backend_lf_plot_metadata(results, backend_label: str, agent_name: str):
    backend_result = result_for_label(results, backend_label)
    backend_series = load_backend_lf_series(results, backend_label, agent_name)
    metadata = common_plot_metadata(
        results,
        plot_type="lf_correlator_backend",
        title=f"{agent_name}: {result_display_label(backend_result)} LF Correlators",
        description="Standalone LF correlator and S-value plot for one backend.",
    )
    metadata["backend_label"] = backend_label
    metadata["backend_display_label"] = result_display_label(backend_result)
    metadata["agent_name"] = agent_name
    metadata["run_count"] = int(backend_series["_run_count"])
    metadata["correlators"] = {
        key: {
            "value": float(backend_series[key]["value"]),
            "error": float(backend_series[key]["stderr"]),
            "ideal_value": float(LF_ANALYTIC_CORRELATORS[key]),
        }
        for key, _, _ in LF_TERM_SPECS
    }
    metadata["s_value"] = {
        "value": float(backend_series["_s_summary"]["value"]),
        "error": float(backend_series["_s_summary"]["stderr"]),
        "ideal_value": float(2.0 * np.sqrt(2.0) - 2.0),
    }
    return metadata


def build_hardware_lf_comparison_metadata(results, agent_name: str):
    fake_result = result_for_label(results, "Fake hardware")
    real_result = result_for_label(results, "Real hardware")
    fake_series = load_backend_lf_series(results, "Fake hardware", agent_name)
    real_series = load_backend_lf_series(results, "Real hardware", agent_name)
    metadata = common_plot_metadata(
        results,
        plot_type="lf_correlator_hardware_comparison",
        title=f"{agent_name}: {result_display_label(fake_result)} vs {result_display_label(real_result)} LF Correlators",
        description="LF correlator comparison between fake-hardware noise simulation and real hardware.",
    )
    metadata["agent_name"] = agent_name
    metadata["ideal_reference"] = {
        "correlators": {key: float(LF_ANALYTIC_CORRELATORS[key]) for key, _, _ in LF_TERM_SPECS},
        "s_value": float(2.0 * np.sqrt(2.0) - 2.0),
    }
    metadata["series"] = [
        {
            "backend_label": "Fake hardware",
            "display_label": result_display_label(fake_result),
            "run_count": int(fake_series["_run_count"]),
            "correlators": {
                key: {
                    "value": float(fake_series[key]["value"]),
                    "error": float(fake_series[key]["stderr"]),
                }
                for key, _, _ in LF_TERM_SPECS
            },
            "s_value": {
                "value": float(fake_series["_s_summary"]["value"]),
                "error": float(fake_series["_s_summary"]["stderr"]),
            },
        },
        {
            "backend_label": "Real hardware",
            "display_label": result_display_label(real_result),
            "run_count": int(real_series["_run_count"]),
            "correlators": {
                key: {
                    "value": float(real_series[key]["value"]),
                    "error": float(real_series[key]["stderr"]),
                }
                for key, _, _ in LF_TERM_SPECS
            },
            "s_value": {
                "value": float(real_series["_s_summary"]["value"]),
                "error": float(real_series["_s_summary"]["stderr"]),
            },
        },
    ]
    return metadata


def build_hardware_lf_agent_summary_metadata(
    results,
    memory_inaccuracy_summary=None,
    *,
    show_epsilon_bounds: bool = False,
):
    real_result = result_for_label(results, "Real hardware")
    agent_rows = []

    for agent_name in HARDWARE_LF_SUMMARY_AGENT_NAMES:
        real_series = load_backend_lf_series(results, "Real hardware", agent_name)
        epsilon = lookup_combined_memory_epsilon(memory_inaccuracy_summary, "Real hardware", agent_name)
        s_summary = real_series["_s_summary"]
        agent_rows.append(
            {
                "agent_name": agent_name,
                "s_value": float(s_summary["value"]),
                "s_error": float(s_summary["stderr"]),
                "run_count": int(real_series["_run_count"]),
                "four_epsilon": None if epsilon is None else float(4.0 * epsilon),
                "four_epsilon_max": four_epsilon_max_from_s_summary(s_summary),
                "correlators": {
                    key: {
                        "value": float(real_series[key]["value"]),
                        "error": float(real_series[key]["stderr"]),
                    }
                    for key, _, _ in LF_TERM_SPECS
                },
            }
        )

    agent_rows.sort(key=lambda row: row["s_value"], reverse=True)
    metadata = common_plot_metadata(
        results,
        plot_type="hardware_lf_agent_summary",
        title=f"{result_display_label(real_result)} LF Violation Comparison Across Agents",
        description=(
            "Real-hardware LF violation comparison across all agents, shown as stacked signed "
            "correlator contributions to S for direct ranking."
        ),
    )
    metadata["backend_label"] = "Real hardware"
    metadata["backend_display_label"] = result_display_label(real_result)
    metadata["show_epsilon_bounds"] = bool(show_epsilon_bounds)
    metadata["ideal_reference"] = {
        "correlators": {key: float(LF_ANALYTIC_CORRELATORS[key]) for key, _, _ in LF_TERM_SPECS},
        "s_value": float(2.0 * np.sqrt(2.0) - 2.0),
    }
    metadata["agents"] = agent_rows
    return metadata


def draw_zero_marker(ax, bar, color, height: float = 0.008):
    ax.hlines(
        y=height,
        xmin=bar.get_x(),
        xmax=bar.get_x() + bar.get_width(),
        colors=color,
        linewidth=3.0,
        zorder=5,
    )


def draw_accuracy_target_blocks(
    ax,
    centers,
    target_values,
    *,
    width: float,
    color: str = ACCURACY_THEORY_LINE_COLOR,
    linewidth: float = 2.0,
):
    return ax.bar(
        centers,
        target_values,
        width=width,
        fill=False,
        edgecolor=color,
        linestyle="--",
        linewidth=linewidth,
        zorder=3,
    )


def annotate_vertical_bars(
    ax,
    bars,
    values,
    *,
    errors=None,
    upper_cap: Optional[float] = None,
    positive_offset: float = 0.015,
    negative_offset: float = 0.055,
    reference_values=None,
    zero_marker_height: float = 0.008,
    fontsize: Optional[int] = None,
):
    if errors is None:
        errors = [0.0] * len(values)
    if reference_values is None:
        reference_values = [None] * len(values)

    for bar, value, error, reference_value in zip(bars, values, errors, reference_values):
        if np.isclose(value, 0.0):
            draw_zero_marker(ax, bar, bar.get_facecolor(), height=zero_marker_height)

        if value >= 0:
            top_reference = value + error
            if reference_value is not None:
                top_reference = max(top_reference, reference_value)
            text_y = top_reference + positive_offset
            va = "bottom"
        else:
            text_y = value - error - negative_offset
            va = "top"

        if upper_cap is not None:
            if text_y > upper_cap:
                text_y = upper_cap
                if value >= 0:
                    va = "top"

        label = f"{value:.3f}" if np.isclose(error, 0.0) else f"{value:.3f}\n± {error:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            label,
            ha="center",
            va=va,
            fontsize=ANNOTATION_FONT_SIZE if fontsize is None else fontsize,
        )


def annotate_custom_bar_labels(ax, bars, values, errors, y_positions, *, vas=None, bbox=None):
    if vas is None:
        vas = ["bottom" if value >= 0 else "top" for value in values]

    for bar, value, error, text_y, va in zip(bars, values, errors, y_positions, vas):
        label = f"{value:.3f}" if np.isclose(error, 0.0) else f"{value:.3f}\n± {error:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            label,
            ha="center",
            va=va,
            fontsize=ANNOTATION_FONT_SIZE,
            bbox=bbox,
            zorder=7,
        )


# -----------------------------------------------------------------------------
# Metric extraction from saved count dictionaries

def extract_strategy_probabilities(counts):
    stats = {
        "0": {"shots": 0, "correct_bets": 0},
        "1": {"shots": 0, "correct_bets": 0},
    }

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 8:
            raise ValueError(f"Bitstring too short to contain M1 and wallet bits: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        c1 = cleaned[M1_BIT_INDEX_FROM_LEFT]
        wallet = cleaned[0:2]
        if c1 not in {"0", "1"}:
            raise ValueError(f"Unexpected M1 value '{c1}' from bitstring '{bitstring}'")
        if wallet not in STAKE_BY_WALLET_STATE:
            raise ValueError(f"Unexpected wallet state '{wallet}' from bitstring '{bitstring}'")

        stake = STAKE_BY_WALLET_STATE[wallet]
        stats[c1]["shots"] += shots
        if c1 == "0" and np.isclose(stake, 1 / 4):
            stats[c1]["correct_bets"] += shots
        if c1 == "1" and np.isclose(stake, 3 / 4):
            stats[c1]["correct_bets"] += shots

    p_bet_1_4_given_c1_0 = stats["0"]["correct_bets"] / stats["0"]["shots"] if stats["0"]["shots"] else 0.0
    p_bet_3_4_given_c1_1 = stats["1"]["correct_bets"] / stats["1"]["shots"] if stats["1"]["shots"] else 0.0
    stderr_c1_0 = np.sqrt(p_bet_1_4_given_c1_0 * (1.0 - p_bet_1_4_given_c1_0) / stats["0"]["shots"]) if stats["0"]["shots"] else 0.0
    stderr_c1_1 = np.sqrt(p_bet_3_4_given_c1_1 * (1.0 - p_bet_3_4_given_c1_1) / stats["1"]["shots"]) if stats["1"]["shots"] else 0.0

    return {
        "P(bet 1/4 | c1=0)": p_bet_1_4_given_c1_0,
        "P(bet 3/4 | c1=1)": p_bet_3_4_given_c1_1,
        "P(bet 1/4 | c1=0) stderr": stderr_c1_0,
        "P(bet 3/4 | c1=1) stderr": stderr_c1_1,
    }


def extract_wallet_counts(counts):
    wallet_counts = {state: 0 for state in PAYOFF_BY_WALLET_STATE}

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 8 or cleaned[-1] == "1":
            continue
        wallet = cleaned[0:2]
        if wallet not in wallet_counts:
            raise ValueError(f"Unexpected wallet state '{wallet}' from bitstring '{bitstring}'")
        wallet_counts[wallet] += shots

    return wallet_counts


def extract_binary_accuracy(counts, *, min_length: int, first_index: int, second_index: int, bit_names: str):
    correct_shots = 0
    total_shots = 0

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < min_length:
            raise ValueError(f"Bitstring too short to contain {bit_names}: {bitstring}")

        # Exclude undo branch: Alice chose A2.
        if cleaned[-1] == "1":
            continue

        total_shots += shots
        if cleaned[first_index] == cleaned[second_index]:
            correct_shots += shots

    accuracy = (correct_shots / total_shots) if total_shots else 0.0
    stderr = np.sqrt(accuracy * (1.0 - accuracy) / total_shots) if total_shots else 0.0
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "correct_shots": correct_shots,
        "total_shots": total_shots,
    }


def extract_guessing_accuracy(counts):
    return extract_binary_accuracy(
        counts,
        min_length=7,
        first_index=GUESSING_G_INDEX_FROM_LEFT,
        second_index=GUESSING_M2_INDEX_FROM_LEFT,
        bit_names="G and M2 bits",
    )


def extract_reflex_accuracy(counts):
    return extract_binary_accuracy(
        counts,
        min_length=6,
        first_index=REFLEX_R_INDEX_FROM_LEFT,
        second_index=REFLEX_M_INDEX_FROM_LEFT,
        bit_names="R and M bits",
    )


def extract_reflex_sa_m_accuracy(counts):
    return extract_binary_accuracy(
        counts,
        min_length=6,
        first_index=REFLEX_M_INDEX_FROM_LEFT,
        second_index=REFLEX_SA_INDEX_FROM_LEFT,
        bit_names="M and S_a bits",
    )


def extract_always_large_accuracy(counts):
    wallet_counts = extract_wallet_counts(counts)
    total_shots = sum(wallet_counts.values())
    correct_shots = sum(
        count for state, count in wallet_counts.items()
        if np.isclose(STAKE_BY_WALLET_STATE[state], 3 / 4)
    )
    accuracy = (correct_shots / total_shots) if total_shots else 0.0
    stderr = np.sqrt(accuracy * (1.0 - accuracy) / total_shots) if total_shots else 0.0
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "correct_shots": correct_shots,
        "total_shots": total_shots,
    }


def expected_payoff_from_wallet_counts(wallet_counts):
    total_shots = sum(wallet_counts.values())
    if total_shots == 0:
        return 0.0
    total_payoff = sum(PAYOFF_BY_WALLET_STATE[state] * count for state, count in wallet_counts.items())
    return total_payoff / total_shots


def expected_payoff_stderr_from_wallet_counts(wallet_counts):
    total_shots = sum(wallet_counts.values())
    if total_shots == 0:
        return 0.0

    payoff_values = np.array([PAYOFF_BY_WALLET_STATE[state] for state in wallet_counts])
    payoff_counts = np.array([wallet_counts[state] for state in wallet_counts])
    probabilities = payoff_counts / total_shots
    mean_payoff = np.sum(probabilities * payoff_values)
    variance = np.sum(probabilities * (payoff_values - mean_payoff) ** 2)
    return np.sqrt(variance / total_shots)


def payoff_stats_from_counts(counts):
    wallet_counts = extract_wallet_counts(counts)
    return {
        "wallet_counts": wallet_counts,
        "payoff": expected_payoff_from_wallet_counts(wallet_counts),
        "payoff_stderr": expected_payoff_stderr_from_wallet_counts(wallet_counts),
    }


def load_lf_violations_for_run(run_dir: Path):
    lf_path = run_dir / "lf_violations" / "lf_violations.json"
    if not lf_path.exists():
        raise FileNotFoundError(f"LF violations file not found: {lf_path.resolve()}")
    return load_json(lf_path)


# -----------------------------------------------------------------------------
# Aggregation and metadata builders

def lf_correlator_series_from_saved_results(agent_lf_data):
    correlators = agent_lf_data["correlators"]
    correlator_shots = agent_lf_data.get("correlator_shots")
    if correlator_shots is None:
        estimated_shots_per_setting = agent_lf_data["total_shots"] / 4.0
        correlator_shots = {key: estimated_shots_per_setting for key in ["E11", "E12", "E21", "E22"]}

    return {
        key: {
            "value": correlators[key],
            "stderr": (
                np.sqrt(max(0.0, 1.0 - correlators[key] ** 2) / correlator_shots[key])
                if correlator_shots[key]
                else 0.0
            ),
            "shots": int(correlator_shots[key]),
        }
        for key in ["E11", "E12", "E21", "E22"]
    }


def aggregate_lf_series(lf_result_paths, agent_name: str):
    per_run_series = []
    per_run_s_values = []

    for lf_result_path in lf_result_paths:
        lf_results = load_json(lf_result_path)
        agent_lf_data = lf_results["agents"][agent_name]
        per_run_series.append(lf_correlator_series_from_saved_results(agent_lf_data))
        per_run_s_values.append(float(agent_lf_data["S"]))

    aggregated = {}
    for key in ["E11", "E12", "E21", "E22"]:
        values = [series[key]["value"] for series in per_run_series]
        single_run_stderr = per_run_series[0][key]["stderr"] if len(per_run_series) == 1 else None
        summary = summarize_measurement(values, single_run_stderr=single_run_stderr)
        aggregated[key] = {
            "value": summary["value"],
            "stderr": summary["stderr"],
            "sigma": summary["sigma"],
            "sem": summary["sem"],
            "shots": int(sum(series[key]["shots"] for series in per_run_series)),
        }

    single_run_s_stderr = None
    if len(per_run_series) == 1:
        single_run_s_stderr = float(np.sqrt(sum(per_run_series[0][key]["stderr"] ** 2 for key in ["E11", "E12", "E21", "E22"])))
    s_summary = summarize_measurement(per_run_s_values, single_run_stderr=single_run_s_stderr)
    aggregated["_s_summary"] = s_summary
    aggregated["_run_count"] = len(per_run_series)
    return aggregated


def load_backend_lf_series(results, backend_label: str, agent_name: str):
    lf_result_paths = result_value(results, backend_label, "lf_result_paths")
    return aggregate_lf_series(lf_result_paths, agent_name)


def agent_label_to_filename(agent_name: str) -> str:
    return "".join(
        ch.lower() if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in agent_name
    ).strip("_")


def theoretical_expected_payoff(win_probability, stake):
    """Expected net payoff for a fair-odds bet paying 1 on success and costing `stake` to enter."""
    return win_probability - stake


def theory_payoff_for_policy(policy_name):
    if policy_name == "betting":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 1 / 4),
            "1": theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "always_small":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 1 / 4),
            "1": theoretical_expected_payoff(3 / 4, 1 / 4),
        }
    elif policy_name == "always_large":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "random":
        per_c1 = {
            "0": 0.5 * theoretical_expected_payoff(1 / 4, 1 / 4) + 0.5 * theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": 0.5 * theoretical_expected_payoff(3 / 4, 1 / 4) + 0.5 * theoretical_expected_payoff(3 / 4, 3 / 4),
        }
    elif policy_name == "opposite":
        per_c1 = {
            "0": theoretical_expected_payoff(1 / 4, 3 / 4),
            "1": theoretical_expected_payoff(3 / 4, 1 / 4),
        }
    else:
        raise ValueError(f"Unknown theoretical policy: {policy_name}")

    return sum(THEORY_C1_PROBABILITIES[c1] * payoff for c1, payoff in per_c1.items())


def aggregate_strategy_probabilities(per_run_results):
    aggregated = {}
    for key in ["P(bet 1/4 | c1=0)", "P(bet 3/4 | c1=1)"]:
        stderr_key = f"{key} stderr"
        values = [result["strategy_probabilities"][key] for result in per_run_results]
        single_run_stderr = per_run_results[0]["strategy_probabilities"][stderr_key] if len(per_run_results) == 1 else None
        summary = summarize_measurement(values, single_run_stderr=single_run_stderr)
        aggregated[key] = summary["value"]
        aggregated[stderr_key] = summary["stderr"]
        aggregated[f"{key} sigma"] = summary["sigma"]
        aggregated[f"{key} sem"] = summary["sem"]
    return aggregated


def extract_backend_run_result(label: str, run_dir: Path, result_filename: str):
    result_path = run_dir / result_filename
    data = load_json(result_path)
    backend_name = infer_backend_name(label, run_dir, data)

    betting_counts = data["agents"]["Betting Agent"]["counts"]
    always_large_counts = data["agents"]["Always 3/4 Agent"]["counts"]
    guessing_counts = data["agents"]["Guessing Agent"]["counts"]
    reflex_counts = data["agents"]["Reflex Agent"]["counts"]

    betting_payoff_stats = payoff_stats_from_counts(betting_counts)
    always_large_payoff_stats = payoff_stats_from_counts(always_large_counts)
    always_large_accuracy_stats = extract_always_large_accuracy(always_large_counts)
    guessing_stats = extract_guessing_accuracy(guessing_counts)
    reflex_stats = extract_reflex_accuracy(reflex_counts)
    reflex_sa_m_stats = extract_reflex_sa_m_accuracy(reflex_counts)

    return {
        "run_dir": run_dir.resolve(),
        "run_name": run_dir.name,
        "result_path": result_path.resolve(),
        "backend_name": backend_name,
        "raw_shots": int(data.get("shots", 0)),
        "strategy_probabilities": extract_strategy_probabilities(betting_counts),
        "observed_payoff": betting_payoff_stats["payoff"],
        "observed_payoff_stderr": betting_payoff_stats["payoff_stderr"],
        "always_large_observed_payoff": always_large_payoff_stats["payoff"],
        "always_large_observed_payoff_stderr": always_large_payoff_stats["payoff_stderr"],
        "always_large_accuracy": always_large_accuracy_stats["accuracy"],
        "always_large_accuracy_stderr": always_large_accuracy_stats["stderr"],
        "always_large_accuracy_shots": always_large_accuracy_stats["total_shots"],
        "guessing_accuracy": guessing_stats["accuracy"],
        "guessing_accuracy_stderr": guessing_stats["stderr"],
        "guessing_accuracy_shots": guessing_stats["total_shots"],
        "reflex_accuracy": reflex_stats["accuracy"],
        "reflex_accuracy_stderr": reflex_stats["stderr"],
        "reflex_accuracy_shots": reflex_stats["total_shots"],
        "reflex_sa_m_accuracy": reflex_sa_m_stats["accuracy"],
        "reflex_sa_m_accuracy_stderr": reflex_sa_m_stats["stderr"],
        "reflex_sa_m_accuracy_shots": reflex_sa_m_stats["total_shots"],
    }


def summarize_scalar_measurement(per_run_results, value_key: str, stderr_key: str):
    values = [result[value_key] for result in per_run_results]
    single_run_stderr = per_run_results[0][stderr_key] if len(per_run_results) == 1 else None
    return summarize_measurement(values, single_run_stderr=single_run_stderr)


def load_backend_result(
    label: str,
    data_dir: Path,
    result_filename: str,
    run_name: Optional[str] = None,
    last_n: Optional[int] = None,
    data_source: str = "paperdata",
):
    run_dirs, selection_mode = resolve_run_dirs(
        label,
        data_dir,
        result_filename,
        run_name=run_name,
        last_n=last_n,
    )
    lf_result_paths = resolve_lf_result_paths(run_dirs)
    per_run_results = [extract_backend_run_result(label, run_dir, result_filename) for run_dir in run_dirs]
    backend_name = summarize_backend_name(result["backend_name"] for result in per_run_results)

    observed_payoff_summary = summarize_scalar_measurement(per_run_results, "observed_payoff", "observed_payoff_stderr")
    always_large_payoff_summary = summarize_scalar_measurement(
        per_run_results,
        "always_large_observed_payoff",
        "always_large_observed_payoff_stderr",
    )
    always_large_accuracy_summary = summarize_scalar_measurement(
        per_run_results,
        "always_large_accuracy",
        "always_large_accuracy_stderr",
    )
    guessing_accuracy_summary = summarize_scalar_measurement(
        per_run_results,
        "guessing_accuracy",
        "guessing_accuracy_stderr",
    )
    reflex_accuracy_summary = summarize_scalar_measurement(
        per_run_results,
        "reflex_accuracy",
        "reflex_accuracy_stderr",
    )
    reflex_sa_m_summary = summarize_scalar_measurement(
        per_run_results,
        "reflex_sa_m_accuracy",
        "reflex_sa_m_accuracy_stderr",
    )

    return {
        "label": label,
        "available": True,
        "backend_name": backend_name,
        "display_label": build_backend_display_label(label, backend_name),
        "axis_label": build_backend_axis_label(label, backend_name),
        "run_dir": run_dirs[0],
        "run_dirs": run_dirs,
        "run_name": run_dirs[0].name if len(run_dirs) == 1 else f"{len(run_dirs)} runs",
        "run_names": [run_dir.name for run_dir in run_dirs],
        "run_count": len(run_dirs),
        "selection_mode": f"{data_source}:{selection_mode}",
        "data_source": data_source,
        "source_result_path": per_run_results[0]["result_path"],
        "source_result_paths": [result["result_path"] for result in per_run_results],
        "raw_shots_per_run": [int(result["raw_shots"]) for result in per_run_results],
        "raw_shots_total": int(sum(result["raw_shots"] for result in per_run_results)),
        "lf_result_paths": lf_result_paths,
        "strategy_probabilities": aggregate_strategy_probabilities(per_run_results),
        "observed_payoff": observed_payoff_summary["value"],
        "observed_payoff_stderr": observed_payoff_summary["stderr"],
        "observed_payoff_sigma": observed_payoff_summary["sigma"],
        "observed_payoff_sem": observed_payoff_summary["sem"],
        "always_large_observed_payoff": always_large_payoff_summary["value"],
        "always_large_observed_payoff_stderr": always_large_payoff_summary["stderr"],
        "always_large_observed_payoff_sigma": always_large_payoff_summary["sigma"],
        "always_large_observed_payoff_sem": always_large_payoff_summary["sem"],
        "always_large_accuracy": always_large_accuracy_summary["value"],
        "always_large_accuracy_stderr": always_large_accuracy_summary["stderr"],
        "always_large_accuracy_sigma": always_large_accuracy_summary["sigma"],
        "always_large_accuracy_sem": always_large_accuracy_summary["sem"],
        "always_large_accuracy_shots": sum(result["always_large_accuracy_shots"] for result in per_run_results),
        "guessing_accuracy": guessing_accuracy_summary["value"],
        "guessing_accuracy_stderr": guessing_accuracy_summary["stderr"],
        "guessing_accuracy_sigma": guessing_accuracy_summary["sigma"],
        "guessing_accuracy_sem": guessing_accuracy_summary["sem"],
        "guessing_accuracy_shots": sum(result["guessing_accuracy_shots"] for result in per_run_results),
        "reflex_accuracy": reflex_accuracy_summary["value"],
        "reflex_accuracy_stderr": reflex_accuracy_summary["stderr"],
        "reflex_accuracy_sigma": reflex_accuracy_summary["sigma"],
        "reflex_accuracy_sem": reflex_accuracy_summary["sem"],
        "reflex_accuracy_shots": sum(result["reflex_accuracy_shots"] for result in per_run_results),
        "reflex_sa_m_accuracy": reflex_sa_m_summary["value"],
        "reflex_sa_m_accuracy_stderr": reflex_sa_m_summary["stderr"],
        "reflex_sa_m_accuracy_sigma": reflex_sa_m_summary["sigma"],
        "reflex_sa_m_accuracy_sem": reflex_sa_m_summary["sem"],
        "reflex_sa_m_accuracy_shots": sum(result["reflex_sa_m_accuracy_shots"] for result in per_run_results),
    }


def build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PLOTS_ROOT / timestamp


# -----------------------------------------------------------------------------
# Accuracy-test extraction

def parse_accuracy_test_agent_label(agent_label: str) -> tuple[str, str]:
    if agent_label.endswith(ACCURACY_TEST_SUFFIX_INIT0):
        return agent_label[:-len(ACCURACY_TEST_SUFFIX_INIT0)], "0"
    if agent_label.endswith(ACCURACY_TEST_SUFFIX_INIT1):
        return agent_label[:-len(ACCURACY_TEST_SUFFIX_INIT1)], "1"
    raise ValueError(f"Unrecognized accuracy-test circuit label: {agent_label}")


def extract_memory_initialization_accuracy(counts, *, expected_bit: str):
    matching_shots = 0
    total_shots = 0

    for bitstring, shots in counts.items():
        cleaned = clean_bitstring(bitstring)
        if len(cleaned) < 3:
            raise ValueError(f"Bitstring too short to contain c[2] memory readout: {bitstring}")

        total_shots += shots
        if cleaned[-3] == expected_bit:
            matching_shots += shots

    accuracy = (matching_shots / total_shots) if total_shots else 0.0
    stderr = np.sqrt(accuracy * (1.0 - accuracy) / total_shots) if total_shots else 0.0
    return {
        "accuracy": accuracy,
        "inaccuracy": 1.0 - accuracy,
        "stderr": stderr,
        "matching_shots": matching_shots,
        "total_shots": total_shots,
        "expected_memory_bit": expected_bit,
    }


def extract_accuracy_test_run_result(label: str, run_dir: Path, result_filename: str):
    result_path = run_dir / result_filename
    if not result_path.exists():
        raise FileNotFoundError(
            f"Accuracy-test result file not found for {label}: {result_path.resolve()}"
        )

    data = load_json(result_path)
    backend_name = infer_backend_name(label, run_dir, data)
    agents = {}

    for agent_label, agent_data in data["agents"].items():
        base_agent_name, expected_bit = parse_accuracy_test_agent_label(agent_label)
        stats = extract_memory_initialization_accuracy(
            agent_data["counts"],
            expected_bit=expected_bit,
        )
        agents[agent_label] = {
            "base_agent_name": base_agent_name,
            **stats,
        }

    return {
        "run_dir": run_dir.resolve(),
        "run_name": run_dir.name,
        "result_path": result_path.resolve(),
        "backend_name": backend_name,
        "raw_shots": int(data.get("shots", 0)),
        "agents": agents,
    }


def summarise_accuracy_test_agent_runs(per_run_agents, *, field_name: str):
    values = [agent_run[field_name] for agent_run in per_run_agents]
    single_run_stderr = per_run_agents[0]["stderr"] if len(per_run_agents) == 1 else None
    summary = summarize_measurement(values, single_run_stderr=single_run_stderr)
    return {
        field_name: summary["value"],
        f"{field_name}_stderr": summary["stderr"],
        f"{field_name}_sigma": summary["sigma"],
        f"{field_name}_sem": summary["sem"],
        "shots": int(sum(agent_run["total_shots"] for agent_run in per_run_agents)),
    }


def load_accuracy_test_backend_result(
    label: str,
    data_dir: Path,
    main_result_filename: str,
    accuracy_test_result_filename: str,
    run_name: Optional[str] = None,
    last_n: Optional[int] = None,
    data_source: str = "paperdata",
):
    run_dirs, selection_mode = resolve_accuracy_test_run_dirs(
        label,
        data_dir,
        main_result_filename,
        accuracy_test_result_filename,
        run_name=run_name,
        last_n=last_n,
    )
    per_run_results = [
        extract_accuracy_test_run_result(label, run_dir, accuracy_test_result_filename)
        for run_dir in run_dirs
    ]
    backend_name = summarize_backend_name(result["backend_name"] for result in per_run_results)

    per_circuit_runs = defaultdict(list)
    for run_result in per_run_results:
        for circuit_label, circuit_result in run_result["agents"].items():
            per_circuit_runs[circuit_label].append(circuit_result)

    circuits = {}
    for circuit_label, circuit_runs in sorted(per_circuit_runs.items()):
        base_agent_name, expected_bit = parse_accuracy_test_agent_label(circuit_label)
        accuracy_summary = summarise_accuracy_test_agent_runs(circuit_runs, field_name="accuracy")
        inaccuracy_summary = summarise_accuracy_test_agent_runs(circuit_runs, field_name="inaccuracy")
        circuits[circuit_label] = {
            "base_agent_name": base_agent_name,
            "expected_memory_bit": expected_bit,
            "accuracy": accuracy_summary["accuracy"],
            "accuracy_stderr": accuracy_summary["accuracy_stderr"],
            "accuracy_sigma": accuracy_summary["accuracy_sigma"],
            "accuracy_sem": accuracy_summary["accuracy_sem"],
            "inaccuracy": inaccuracy_summary["inaccuracy"],
            "inaccuracy_stderr": inaccuracy_summary["inaccuracy_stderr"],
            "inaccuracy_sigma": inaccuracy_summary["inaccuracy_sigma"],
            "inaccuracy_sem": inaccuracy_summary["inaccuracy_sem"],
            "shots": accuracy_summary["shots"],
        }

    return {
        "label": label,
        "backend_name": backend_name,
        "display_label": build_backend_display_label(label, backend_name),
        "axis_label": build_backend_axis_label(label, backend_name),
        "run_dir": run_dirs[0],
        "run_dirs": run_dirs,
        "run_name": run_dirs[0].name if len(run_dirs) == 1 else f"{len(run_dirs)} runs",
        "run_names": [run_dir.name for run_dir in run_dirs],
        "run_count": len(run_dirs),
        "selection_mode": f"{data_source}:{selection_mode}",
        "data_source": data_source,
        "source_result_paths": [result["result_path"] for result in per_run_results],
        "raw_shots_per_run": [int(result["raw_shots"]) for result in per_run_results],
        "raw_shots_total": int(sum(result["raw_shots"] for result in per_run_results)),
        "circuits": circuits,
    }


def build_memory_inaccuracy_summary(results) -> dict:
    def combine_backend_circuits(result):
        circuits = result.get("circuits", {})
        combined = {}
        for base_agent_name in STANDARD_AGENT_NAMES:
            init0_label = f"{base_agent_name}{ACCURACY_TEST_SUFFIX_INIT0}"
            init1_label = f"{base_agent_name}{ACCURACY_TEST_SUFFIX_INIT1}"
            if init0_label not in circuits or init1_label not in circuits:
                continue

            init0 = circuits[init0_label]
            init1 = circuits[init1_label]
            combined_accuracy = 0.5 * (init0["accuracy"] + init1["accuracy"])
            combined_inaccuracy = 0.5 * (init0["inaccuracy"] + init1["inaccuracy"])
            combined_accuracy_stderr = 0.5 * np.sqrt(
                init0["accuracy_stderr"] ** 2 + init1["accuracy_stderr"] ** 2
            )
            combined_inaccuracy_stderr = 0.5 * np.sqrt(
                init0["inaccuracy_stderr"] ** 2 + init1["inaccuracy_stderr"] ** 2
            )
            combined[base_agent_name] = {
                "mean_init_accuracy": combined_accuracy,
                "mean_init_accuracy_stderr": combined_accuracy_stderr,
                "epsilon": combined_inaccuracy,
                "epsilon_stderr": combined_inaccuracy_stderr,
                "init0_accuracy": init0["accuracy"],
                "init0_accuracy_stderr": init0["accuracy_stderr"],
                "init1_accuracy": init1["accuracy"],
                "init1_accuracy_stderr": init1["accuracy_stderr"],
                "shots": int(init0["shots"] + init1["shots"]),
            }
        return combined

    selected_inputs = []
    for result in results:
        selected_inputs.append({
            "label": result["label"],
            "display_label": result["display_label"],
            "backend_name": result["backend_name"],
            "selection_mode": result["selection_mode"],
            "run_count": int(result["run_count"]),
            "run_names": list(result["run_names"]),
            "run_dirs": [str(path) for path in result["run_dirs"]],
            "source_result_paths": [str(path) for path in result.get("source_result_paths", [])],
            "raw_shots_per_run": [int(value) for value in result.get("raw_shots_per_run", [])],
            "raw_shots_total": int(result.get("raw_shots_total", 0)),
            "available": bool(result.get("available", True)),
            "error": result.get("error"),
        })

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Memory inaccuracy for the hard-coded accuracy-test circuits. "
            "For each circuit, the initialized M bit is compared against the final measurement of c[2]."
        ),
        "selected_inputs": selected_inputs,
        "backends": [
            {
                "label": result["label"],
                "display_label": result["display_label"],
                "axis_label": result["axis_label"],
                "backend_name": result["backend_name"],
                "available": bool(result.get("available", True)),
                "error": result.get("error"),
                "selection_mode": result["selection_mode"],
                "run_count": int(result["run_count"]),
                "run_names": list(result["run_names"]),
                "source_result_paths": [str(path) for path in result.get("source_result_paths", [])],
                "circuits": result.get("circuits", {}),
                "combined_agents": combine_backend_circuits(result),
            }
            for result in results
        ],
    }


def missing_accuracy_test_result(label: str, main_result: dict, error: FileNotFoundError) -> dict:
    return {
        "label": label,
        "available": False,
        "backend_name": main_result["backend_name"],
        "display_label": main_result["display_label"],
        "axis_label": main_result["axis_label"],
        "run_dir": main_result["run_dir"],
        "run_dirs": main_result["run_dirs"],
        "run_name": main_result["run_name"],
        "run_names": main_result["run_names"],
        "run_count": main_result["run_count"],
        "selection_mode": f"{main_result['selection_mode']}_missing_accuracy_test_data",
        "source_result_paths": [],
        "raw_shots_per_run": [],
        "raw_shots_total": 0,
        "agents": {},
        "circuits": {},
        "error": str(error),
    }


def lookup_combined_memory_epsilon(memory_inaccuracy_summary, backend_label: str, agent_name: str) -> Optional[float]:
    if memory_inaccuracy_summary is None:
        return None

    backend_result = next(
        (
            backend
            for backend in memory_inaccuracy_summary.get("backends", [])
            if backend.get("label") == backend_label
        ),
        None,
    )
    if backend_result is None:
        return None

    combined = backend_result.get("combined_agents", {}).get(agent_name)
    if combined is None:
        return None

    epsilon = combined.get("epsilon")
    if epsilon is None or not np.isfinite(epsilon):
        return None
    return float(epsilon)


def four_epsilon_max_from_s_summary(s_summary: dict) -> float:
    return float(s_summary["value"] - 3.0 * s_summary["stderr"])


def build_tracking_epsilon_max_summary(results, backend_label: str = "Real hardware") -> dict:
    backend_result = result_for_label(results, backend_label)
    agents = {}

    for agent_name in MEMORY_PLOT_AGENT_NAMES:
        series = load_backend_lf_series(results, backend_label, agent_name)
        s_summary = series["_s_summary"]
        s_obs = float(s_summary["value"])
        sigma_l = float(s_summary["stderr"])
        epsilon_max = four_epsilon_max_from_s_summary(s_summary) / 4.0

        agents[agent_name] = {
            "s_obs": s_obs,
            "sigma_l": sigma_l,
            "epsilon_max": float(epsilon_max),
            "run_count": int(series["_run_count"]),
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Maximum allowed tracking error epsilon inferred from the aggregated LF "
            "violation summaries. Values are computed from the selected runs using "
            "epsilon_max = (S_obs - 3*sigma)/4."
        ),
        "backend_label": backend_label,
        "backend_display_label": result_display_label(backend_result),
        "selection_mode": backend_result["selection_mode"],
        "run_count": int(backend_result["run_count"]),
        "run_names": list(backend_result["run_names"]),
        "criterion": "S_obs - 3*sigma > 4*epsilon",
        "agents": agents,
    }


def build_payoff_value_summary(results) -> dict:
    theory_values = {
        "betting_agent": float(theory_payoff_for_policy("betting")),
        "always_3_4_agent": float(theory_payoff_for_policy("always_large")),
    }
    backends = []

    for result in results:
        betting_observed = float(result["observed_payoff"])
        always_large_observed = float(result["always_large_observed_payoff"])
        backends.append(
            {
                "label": result["label"],
                "display_label": result_display_label(result),
                "axis_label": result_axis_label(result),
                "backend_name": result.get("backend_name"),
                "selection_mode": result.get("selection_mode"),
                "run_count": int(result.get("run_count", 0)),
                "run_names": list(result.get("run_names", [])),
                "raw_shots_total": int(result.get("raw_shots_total", 0)),
                "betting_agent": {
                    "observed": betting_observed,
                    "stderr": float(result["observed_payoff_stderr"]),
                    "theory": theory_values["betting_agent"],
                    "observed_minus_theory": betting_observed - theory_values["betting_agent"],
                },
                "always_3_4_agent": {
                    "observed": always_large_observed,
                    "stderr": float(result["always_large_observed_payoff_stderr"]),
                    "theory": theory_values["always_3_4_agent"],
                    "observed_minus_theory": always_large_observed - theory_values["always_3_4_agent"],
                },
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Observed payoff values and theoretical references used in the payoff "
            "comparison plot. Full floating-point values are saved here so tiny "
            "deviations that are visually hidden in the figure remain accessible."
        ),
        "backends": backends,
    }


# -----------------------------------------------------------------------------
# Plot builders

def plot_combined_memory_epsilon(memory_inaccuracy_summary, output_dir: Path, tracking_epsilon_max_summary: Optional[dict] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_names = MEMORY_PLOT_AGENT_NAMES
    plotted_backend_labels = ["Fake hardware", "Real hardware"]
    x = np.arange(len(agent_names))
    width = 0.28
    tracking_box_width = 0.78

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    max_height = 0.0

    for idx, backend_label in enumerate(plotted_backend_labels):
        backend_result = next(
            backend for backend in memory_inaccuracy_summary["backends"]
            if backend["label"] == backend_label
        )
        combined_agents = backend_result["combined_agents"]
        values = [combined_agents.get(agent, {}).get("epsilon", np.nan) for agent in agent_names]
        errors = [combined_agents.get(agent, {}).get("epsilon_stderr", np.nan) for agent in agent_names]
        offsets = x + (idx - (len(plotted_backend_labels) - 1) / 2.0) * width
        bars = ax.bar(
            offsets,
            values,
            width=width,
            yerr=errors,
            capsize=5,
            ecolor="#333333",
            color=MEMORY_BACKEND_COLORS[backend_label],
            edgecolor="black",
            linewidth=1.0,
            label=backend_result["axis_label"],
        )
        for value, error in zip(values, errors):
            if np.isfinite(value) and np.isfinite(error):
                max_height = max(max_height, value + error)
    if tracking_epsilon_max_summary is not None:
        for center, agent_name in zip(x, agent_names):
            agent_summary = tracking_epsilon_max_summary.get("agents", {}).get(agent_name)
            if agent_summary is None:
                continue
            epsilon_max = agent_summary.get("epsilon_max")
            if epsilon_max is None or not np.isfinite(epsilon_max) or epsilon_max <= 0.0:
                continue
            draw_accuracy_target_blocks(
                ax,
                [center],
                [float(epsilon_max)],
                width=tracking_box_width,
                color=ACCURACY_THEORY_LINE_COLOR,
                linewidth=1.8,
            )
            max_height = max(max_height, float(epsilon_max))

    ax.set_xticks(x)
    set_thesis_xticklabels(ax, [agent_display_label(agent_name) for agent_name in agent_names])
    ax.set_ylim(0.0, 0.0225)
    style_memory_axes(ax, r"$P(c \neq a \mid x=1)$")
    legend_handles = [
        Patch(
            facecolor=MEMORY_BACKEND_COLORS[backend_label],
            edgecolor="black",
            linewidth=1.0,
            label=next(
                backend["axis_label"]
                for backend in memory_inaccuracy_summary["backends"]
                if backend["label"] == backend_label
            ),
        )
        for backend_label in plotted_backend_labels
    ]
    if tracking_epsilon_max_summary is not None:
        legend_handles.append(
            Line2D([], [], color=ACCURACY_THEORY_LINE_COLOR, linestyle="--", linewidth=1.8, label=r"Maximum allowed $\epsilon$")
        )
    place_legend_above_axes(
        fig,
        ax,
        ncol=3 if tracking_epsilon_max_summary is not None else 2,
        fontsize=LEGEND_FONT_SIZE,
        handles=legend_handles,
    )

    plot_path = output_dir / "combined_memory_initialization_epsilon_comparison.png"
    save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_born_rule_accuracy(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [r"$P(\mathrm{bet}\;1/4\mid c_1=0)$", r"$P(\mathrm{bet}\;3/4\mid c_1=1)$"]
    x = np.arange(len(categories))
    width = 0.18
    group_spacing = 0.26
    target_width = (len(results) - 1) * group_spacing + width + 0.12
    y_max = 1.2

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    draw_accuracy_target_blocks(
        ax,
        x,
        [1.0] * len(categories),
        width=target_width,
    )
    for idx, result in enumerate(results):
        values = [
            result["strategy_probabilities"]["P(bet 1/4 | c1=0)"],
            result["strategy_probabilities"]["P(bet 3/4 | c1=1)"],
        ]
        errors = [
            result["strategy_probabilities"]["P(bet 1/4 | c1=0) stderr"],
            result["strategy_probabilities"]["P(bet 3/4 | c1=1) stderr"],
        ]
        offset = (idx - (len(results) - 1) / 2) * group_spacing
        centers = x + offset
        bars = ax.bar(
            centers,
            values,
            yerr=errors,
            capsize=5,
            ecolor="#333333",
            width=width,
            label=result_display_label(result),
            color=ACCURACY_BACKEND_COLORS[result["label"]],
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        annotate_vertical_bars(
            ax,
            bars,
            values,
            errors=errors,
            upper_cap=y_max - 0.02,
            positive_offset=0.02,
            reference_values=[1.0] * len(values),
            fontsize=ANNOTATION_FONT_SIZE,
        )

    ax.set_xticks(x)
    set_thesis_xticklabels(ax, categories)
    ax.set_ylim(0.0, y_max)
    style_accuracy_axes(ax, "Probability")
    set_probability_axis_ticks(ax, y_max)
    legend_handles = [
        Patch(
            facecolor=ACCURACY_BACKEND_COLORS[result["label"]],
            edgecolor="black",
            linewidth=1.0,
            label=result_display_label(result),
        )
        for result in results
    ]
    legend_handles.append(
        Line2D([], [], color=ACCURACY_THEORY_LINE_COLOR, linestyle="--", linewidth=2.0, label="Ideal target")
    )
    place_legend_above_axes(fig, ax, ncol=2, fontsize=LEGEND_FONT_SIZE, handles=legend_handles)

    plot_path = output_dir / "betting_agent_accuracy_comparison.png"
    save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_always_large_vs_betting_payoff_comparison(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_labels = [result_axis_label(result_for_label(results, label)) for label in BACKEND_LABELS]
    x = np.arange(len(backend_labels))
    width = 0.34

    born_rule_values = backend_values(results, "observed_payoff")
    born_rule_errors = backend_values(results, "observed_payoff_stderr")
    always_large_values = backend_values(results, "always_large_observed_payoff")
    always_large_errors = backend_values(results, "always_large_observed_payoff_stderr")
    born_rule_theory = theory_payoff_for_policy("betting")
    always_large_theory = theory_payoff_for_policy("always_large")

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    born_rule_bars = ax.bar(
        x - width / 2,
        born_rule_values,
        yerr=born_rule_errors,
        capsize=4,
        ecolor="#333333",
        width=width,
        color=THEORY_COMPARISON_COLORS["Born-rule"],
        edgecolor="black",
        linewidth=1.0,
        label="Betting Agent",
    )
    always_large_bars = ax.bar(
        x + width / 2,
        always_large_values,
        yerr=always_large_errors,
        capsize=4,
        ecolor="#333333",
        width=width,
        color=THEORY_COMPARISON_COLORS["Always 3/4"],
        edgecolor="black",
        linewidth=1.0,
        label="Always-3/4 agent",
    )
    annotate_vertical_bars(
        ax,
        born_rule_bars,
        born_rule_values,
        errors=born_rule_errors,
        positive_offset=0.015,
        negative_offset=0.015,
    )
    always_large_text_y = []
    always_large_text_va = []
    for value, error in zip(always_large_values, always_large_errors):
        if value >= 0:
            always_large_text_y.append(value + error + 0.015)
            always_large_text_va.append("bottom")
        else:
            always_large_text_y.append(value - error - 0.015)
            always_large_text_va.append("top")
    annotate_custom_bar_labels(
        ax,
        always_large_bars,
        always_large_values,
        always_large_errors,
        always_large_text_y,
        vas=always_large_text_va,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0},
    )

    for bar in born_rule_bars:
        ax.hlines(
            y=born_rule_theory,
            xmin=bar.get_x() + 0.04,
            xmax=bar.get_x() + bar.get_width() - 0.04,
            colors="#C92A2A",
            linestyles="--",
            linewidth=1.8,
            zorder=4,
        )
    ax.axhline(always_large_theory, color="#C92A2A", linestyle="--", linewidth=1.8, zorder=4)

    ax.axhline(0.0, color=IDEAL_COLOR, linewidth=1.0)
    ax.set_xticks(x)
    set_thesis_xticklabels(ax, backend_labels)
    ax.set_ylim(-0.33, 0.12)
    style_bar_axes(ax, "Betting Agent vs Always-3/4 Payoff", "Expected payoff")
    place_legend_above_axes(
        fig,
        ax,
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        handles=[
            Patch(facecolor=THEORY_COMPARISON_COLORS["Born-rule"], edgecolor="black", label="Betting Agent"),
            Patch(facecolor=THEORY_COMPARISON_COLORS["Always 3/4"], edgecolor="black", label="Always-3/4 agent"),
            Line2D([0], [0], color="#C92A2A", linestyle="--", linewidth=1.8, label="Theoretical value"),
        ],
    )

    plot_path = output_dir / "betting_agent_vs_always_3_4_payoff_comparison.png"
    save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_backend_lf_correlator_comparisons(results, output_dir: Path, backend_label: str, memory_inaccuracy_summary=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_filename_prefix = backend_label.lower().replace(" ", "_")
    backend_result = result_for_label(results, backend_label)
    backend_title_prefix = result_display_label(backend_result)
    lower_panel_label = backend_lf_panel_label(backend_result)

    saved_paths = []
    classical_bound = 2.0
    tsirelson_bound = 2.0 * np.sqrt(2.0)
    violation_offset = -classical_bound
    tsirelson_violation = tsirelson_bound - classical_bound

    for agent_name in LF_AGENT_NAMES:
        backend_series = load_backend_lf_series(results, backend_label, agent_name)
        epsilon = lookup_combined_memory_epsilon(memory_inaccuracy_summary, backend_label, agent_name)
        four_epsilon = None if epsilon is None else 4.0 * epsilon
        s_summary = backend_series["_s_summary"]
        run_count = backend_series["_run_count"]

        raw_values = np.array([backend_series[key]["value"] for key, _, _ in LF_TERM_SPECS])
        raw_errors = np.array([backend_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        raw_theory_values = np.array([LF_ANALYTIC_CORRELATORS[key] for key, _, _ in LF_TERM_SPECS])
        signed_term_values = np.array([sign * backend_series[key]["value"] for key, sign, _ in LF_TERM_SPECS])
        signed_term_errors = np.array([backend_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        signed_term_theory_values = np.array([sign * LF_ANALYTIC_CORRELATORS[key] for key, sign, _ in LF_TERM_SPECS])
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=LF_SINGLE_BACKEND_FIGSIZE,
            gridspec_kw={"height_ratios": [1.8, 1], "hspace": 0.15},
        )

        x_pos = np.arange(len(LF_TERM_SPECS))
        bar_width = 0.8

        for idx in range(len(LF_TERM_SPECS)):
            ax1.bar(
                x_pos[idx],
                raw_values[idx],
                width=bar_width,
                color=LF_TERM_COLORS[idx],
                edgecolor="black",
                yerr=raw_errors[idx],
                capsize=5,
                zorder=2,
            )
            ax1.bar(
                x_pos[idx],
                raw_theory_values[idx],
                width=bar_width,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2,
                zorder=3,
            )

        ax1.set_xticks(x_pos)
        set_thesis_xticklabels(ax1, LF_CORRELATOR_LABELS)
        ax1.set_xlim(-0.8, len(LF_TERM_SPECS) - 0.2)
        ax1.set_ylim(-0.8, 0.8)
        ax1.set_ylabel("Correlator value")
        apply_thesis_axis_typography(ax1)
        ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax1.grid(axis="y", alpha=0.25)
        ax1.plot([], [], color="red", linestyle="--", linewidth=2, label="Ideal theoretical value")
        ax1.plot([], [], color="black", linewidth=1.5, marker="|", markersize=10, label="Standard error of the mean (SEM)")
        ax1.legend(loc="upper right", fontsize=LF_LEGEND_FONT_SIZE, frameon=True)

        left_exp = violation_offset
        left_th = violation_offset
        bar_height = 0.5
        cumulative_centers = []
        cumulative_errors = []
        cumulative_variance = 0.0

        for idx in range(len(LF_TERM_SPECS)):
            width_exp = abs(signed_term_values[idx])
            width_th = abs(signed_term_theory_values[idx])
            term_sigma = signed_term_errors[idx]
            theory_height = bar_height + 0.1

            ax2.barh(
                0,
                width_exp,
                height=bar_height,
                left=left_exp,
                color=LF_TERM_COLORS[idx],
                edgecolor="none",
                zorder=2,
            )

            theory_left = left_th
            theory_right = left_th + width_th
            theory_ymin = -theory_height / 2.0
            theory_ymax = theory_height / 2.0
            ax2.hlines(
                [theory_ymin, theory_ymax],
                theory_left,
                theory_right,
                colors="red",
                linestyles="--",
                linewidth=1.5,
                zorder=3,
            )
            ax2.vlines(
                [theory_left, theory_right],
                theory_ymin,
                theory_ymax,
                colors="red",
                linestyles="--",
                linewidth=1.5,
                zorder=3,
            )
            left_exp += width_exp
            left_th += width_th

            cumulative_variance += term_sigma ** 2
            cumulative_centers.append(left_exp)
            cumulative_errors.append(np.sqrt(cumulative_variance))

        threshold_ymin = -(bar_height + 0.28) / 2.0
        threshold_ymax = (bar_height + 0.28) / 2.0
        for threshold in [0.0, tsirelson_violation]:
            ax2.vlines(
                threshold,
                threshold_ymin,
                threshold_ymax,
                colors="red",
                linewidth=2.0,
                zorder=5,
            )
        if SHOW_FOUR_EPSILON_IN_BACKEND_LF_PLOTS and four_epsilon is not None:
            line_top_y = threshold_ymax
            ax2.vlines(
                four_epsilon,
                threshold_ymin,
                threshold_ymax,
                colors="black",
                linewidth=2.0,
                zorder=6,
            )
            ax2.text(
                four_epsilon + 0.015,
                line_top_y,
                f"$4\\epsilon$ = {four_epsilon:.3f}",
                ha="left",
                va="top",
                fontsize=LF_VALUE_LABEL_FONT_SIZE,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
            )

        final_sigma = s_summary["stderr"]
        final_violation = s_summary["value"]
        if run_count > 1:
            ax2.errorbar(
                [cumulative_centers[-1]],
                [0.0],
                xerr=[final_sigma / 2.0],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )
        else:
            ax2.errorbar(
                cumulative_centers,
                [0] * len(cumulative_centers),
                xerr=[err / 2.0 for err in cumulative_errors],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )
        final_text_x = cumulative_centers[-1] + final_sigma / 2.0 + 0.08
        ax2.text(
            final_text_x,
            0.0,
            f"S = {final_violation:.3f}\n± {final_sigma:.3f}",
            ha="left",
            va="center",
            fontsize=LF_VALUE_LABEL_FONT_SIZE,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        )

        right_limit = max(tsirelson_violation + 0.12, final_text_x + 0.42)
        if SHOW_FOUR_EPSILON_IN_BACKEND_LF_PLOTS and four_epsilon is not None:
            right_limit = max(right_limit, four_epsilon + 0.12)
        ax2.set_xlim(violation_offset, right_limit)
        ax2.set_ylim(-1, 1)
        ax2.set_yticks([])
        ax2.text(
            -0.09,
            0.0,
            lower_panel_label,
            transform=ax2.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=LF_PANEL_LABEL_FONT_SIZE,
            clip_on=False,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        )
        for spine in ["top", "left", "right"]:
            ax2.spines[spine].set_visible(False)

        ax2.set_xticks([violation_offset, 0.0, tsirelson_violation])
        set_thesis_xticklabels(ax2, ["-2", "0", r"$2\sqrt{2}-2$"], fontsize=LF_LOWER_TICK_FONT_SIZE)
        ax2.spines["bottom"].set_linewidth(1.5)

        fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.12, hspace=0.18)

        plot_path = output_dir / f"{backend_filename_prefix}_{agent_label_to_filename(agent_name)}_lf_correlator_comparison.png"
        save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def plot_hardware_lf_comparison_per_agent(results, output_dir: Path, memory_inaccuracy_summary=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    classical_bound = 2.0
    tsirelson_bound = 2.0 * np.sqrt(2.0)
    violation_offset = -classical_bound
    tsirelson_violation = tsirelson_bound - classical_bound
    fake_result = result_for_label(results, "Fake hardware")
    real_result = result_for_label(results, "Real hardware")
    fake_label = result_display_label(fake_result)
    real_label = result_display_label(real_result)

    for agent_name in LF_AGENT_NAMES:
        fake_series = load_backend_lf_series(results, "Fake hardware", agent_name)
        real_series = load_backend_lf_series(results, "Real hardware", agent_name)
        real_epsilon = lookup_combined_memory_epsilon(memory_inaccuracy_summary, "Real hardware", agent_name)
        fake_s_summary = fake_series["_s_summary"]
        real_s_summary = real_series["_s_summary"]
        fake_run_count = fake_series["_run_count"]
        real_run_count = real_series["_run_count"]

        x_pos = np.arange(len(LF_TERM_SPECS))
        fake_values = np.array([fake_series[key]["value"] for key, _, _ in LF_TERM_SPECS])
        fake_errors = np.array([fake_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        real_values = np.array([real_series[key]["value"] for key, _, _ in LF_TERM_SPECS])
        real_errors = np.array([real_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        ideal_values = np.array([LF_ANALYTIC_CORRELATORS[key] for key, _, _ in LF_TERM_SPECS])

        signed_fake_values = np.array([sign * fake_series[key]["value"] for key, sign, _ in LF_TERM_SPECS])
        signed_fake_errors = np.array([fake_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        signed_real_values = np.array([sign * real_series[key]["value"] for key, sign, _ in LF_TERM_SPECS])
        signed_real_errors = np.array([real_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS])
        signed_ideal_values = np.array([sign * LF_ANALYTIC_CORRELATORS[key] for key, sign, _ in LF_TERM_SPECS])
        signed_term_labels = [label for _, _, label in LF_TERM_SPECS]

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=LF_COMPARISON_FIGSIZE,
            gridspec_kw={"height_ratios": [1.8, 1.3], "hspace": 0.18},
        )

        outline_width = 0.78
        bar_width = 0.26
        fake_offset = -0.16
        real_offset = 0.16

        for idx in range(len(LF_TERM_SPECS)):
            ax1.bar(
                x_pos[idx] + fake_offset,
                fake_values[idx],
                width=bar_width,
                color=LF_TERM_COLORS[idx],
                edgecolor="black",
                linestyle="--",
                yerr=fake_errors[idx],
                capsize=5,
                linewidth=1.4,
                zorder=2,
            )
            ax1.bar(
                x_pos[idx] + real_offset,
                real_values[idx],
                width=bar_width,
                color=LF_TERM_COLORS[idx],
                edgecolor="black",
                yerr=real_errors[idx],
                capsize=5,
                linewidth=1.0,
                zorder=2,
            )
            ax1.bar(
                x_pos[idx],
                ideal_values[idx],
                width=outline_width,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2,
                zorder=3,
            )

        ax1.set_xticks(x_pos)
        set_thesis_xticklabels(ax1, LF_CORRELATOR_LABELS)
        ax1.set_xlim(-0.8, len(LF_TERM_SPECS) - 0.2)
        ax1.set_ylim(-0.8, 0.8)
        ax1.set_ylabel("Correlator value")
        apply_thesis_axis_typography(ax1)
        ax1.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax1.grid(axis="y", alpha=0.25)
        hardware_handles = [
            Patch(facecolor="white", edgecolor="black", linewidth=1.0, label=real_label),
            Patch(facecolor="white", edgecolor="black", linewidth=1.4, linestyle="--", label=fake_label),
            Line2D([], [], color="red", linestyle="--", linewidth=2, label="Ideal theoretical value"),
            Line2D([], [], color="black", linewidth=1.5, marker="|", markersize=10, label="Standard error of the mean (SEM)"),
        ]
        ax1.legend(handles=hardware_handles, loc="upper right", fontsize=LF_LEGEND_FONT_SIZE, frameon=True)

        row_specs = [
            (fake_label, 0.22, signed_fake_values, signed_fake_errors, fake_s_summary, fake_run_count),
            (real_label, -0.22, signed_real_values, signed_real_errors, real_s_summary, real_run_count),
        ]
        bar_height = 0.26
        max_right_limit = tsirelson_violation + 0.12

        for row_label, y_pos, signed_values, signed_errors, s_summary, row_run_count in row_specs:
            left_exp = violation_offset
            left_ideal = violation_offset
            cumulative_centers = []
            cumulative_errors = []
            cumulative_variance = 0.0

            for idx, label in enumerate(signed_term_labels):
                width_exp = abs(signed_values[idx])
                width_ideal = abs(signed_ideal_values[idx])
                theory_height = bar_height + 0.08

                ax2.barh(
                    y_pos,
                    width_exp,
                    height=bar_height,
                    left=left_exp,
                    color=LF_TERM_COLORS[idx],
                    edgecolor="none",
                    zorder=2,
                )

                theory_left = left_ideal
                theory_right = left_ideal + width_ideal
                theory_ymin = y_pos - theory_height / 2.0
                theory_ymax = y_pos + theory_height / 2.0
                ax2.hlines(
                    [theory_ymin, theory_ymax],
                    theory_left,
                    theory_right,
                    colors="red",
                    linestyles="--",
                    linewidth=1.5,
                    zorder=3,
                )
                ax2.vlines(
                    [theory_left, theory_right],
                    theory_ymin,
                    theory_ymax,
                    colors="red",
                    linestyles="--",
                    linewidth=1.5,
                    zorder=3,
                )
                left_exp += width_exp
                left_ideal += width_ideal
                cumulative_variance += signed_errors[idx] ** 2
                cumulative_centers.append(left_exp)
                cumulative_errors.append(np.sqrt(cumulative_variance))

            final_sigma = s_summary["stderr"]
            final_violation = s_summary["value"]
            if row_run_count > 1:
                ax2.errorbar(
                    [cumulative_centers[-1]],
                    [y_pos],
                    xerr=[final_sigma / 2.0],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=4,
                )
            else:
                ax2.errorbar(
                    cumulative_centers,
                    [y_pos] * len(cumulative_centers),
                    xerr=[err / 2.0 for err in cumulative_errors],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=4,
                )
            final_text_x = cumulative_centers[-1] + final_sigma / 2.0 + 0.08
            ax2.text(
                final_text_x,
                y_pos,
                f"S = {final_violation:.3f}\n± {final_sigma:.3f}",
                ha="left",
                va="center",
                fontsize=LF_VALUE_LABEL_FONT_SIZE,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            )
            max_right_limit = max(max_right_limit, final_text_x + 0.34)

        threshold_ymin = 0.0
        threshold_ymax = 0.97
        for threshold in [0.0, tsirelson_violation]:
            ax2.axvline(
                x=threshold,
                color="red",
                linewidth=2.2,
                ymin=threshold_ymin,
                ymax=threshold_ymax,
                zorder=5,
            )
        if SHOW_FOUR_EPSILON_IN_HARDWARE_LF_COMPARISON_PLOTS and real_epsilon is not None:
            four_epsilon = 4.0 * real_epsilon
            line_top_y = -0.55 + threshold_ymax * (0.55 - (-0.55))
            ax2.axvline(
                x=four_epsilon,
                color="black",
                linewidth=2.0,
                ymin=threshold_ymin,
                ymax=threshold_ymax,
                zorder=6,
            )
            ax2.text(
                four_epsilon + 0.015,
                line_top_y,
                f"$4\\epsilon$ = {four_epsilon:.3f}",
                ha="left",
                va="top",
                fontsize=LF_VALUE_LABEL_FONT_SIZE,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
            )
            max_right_limit = max(max_right_limit, four_epsilon + 0.14)

        ax2.set_xlim(violation_offset, max_right_limit)
        ax2.set_ylim(-0.55, 0.55)
        ax2.set_yticks([])
        ax2.text(
            -0.09,
            0.22,
            "Noise\nsimulation",
            transform=ax2.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=LF_PANEL_LABEL_FONT_SIZE,
            multialignment="center",
            clip_on=False,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        )
        ax2.text(
            -0.09,
            -0.22,
            "Hardware",
            transform=ax2.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=LF_PANEL_LABEL_FONT_SIZE,
            clip_on=False,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        )
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.set_xticks([violation_offset, 0.0, tsirelson_violation])
        set_thesis_xticklabels(ax2, ["-2", "0", r"$2\sqrt{2}-2$"], fontsize=LF_LOWER_TICK_FONT_SIZE)
        ax2.spines["bottom"].set_linewidth(1.5)

        fig.subplots_adjust(left=0.22, right=0.992, top=0.92, bottom=0.10, hspace=0.18)

        plot_path = output_dir / f"hardware_comparison_{agent_label_to_filename(agent_name)}_lf_correlator_comparison.png"
        save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def plot_hardware_lf_agent_summary(
    results,
    output_dir: Path,
    memory_inaccuracy_summary=None,
    *,
    show_epsilon_bounds: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_bound = 2.0
    tsirelson_bound = 2.0 * np.sqrt(2.0)
    violation_offset = -classical_bound
    tsirelson_violation = tsirelson_bound - classical_bound
    real_result = result_for_label(results, "Real hardware")
    real_label = result_display_label(real_result)

    agent_rows = []
    for agent_name in HARDWARE_LF_SUMMARY_AGENT_NAMES:
        real_series = load_backend_lf_series(results, "Real hardware", agent_name)
        epsilon = lookup_combined_memory_epsilon(memory_inaccuracy_summary, "Real hardware", agent_name)
        agent_rows.append(
            {
                "agent_name": agent_name,
                "series": real_series,
                "signed_values": np.array([sign * real_series[key]["value"] for key, sign, _ in LF_TERM_SPECS]),
                "signed_errors": np.array([real_series[key]["stderr"] for key, _, _ in LF_TERM_SPECS]),
                "four_epsilon": None if epsilon is None else 4.0 * epsilon,
                "four_epsilon_max": four_epsilon_max_from_s_summary(real_series["_s_summary"]),
            }
        )

    agent_rows.sort(key=lambda row: row["series"]["_s_summary"]["value"], reverse=True)
    signed_ideal_values = np.array([sign * LF_ANALYTIC_CORRELATORS[key] for key, sign, _ in LF_TERM_SPECS])

    fig_height = 4.8 + 0.75 * len(agent_rows)
    fig, ax = plt.subplots(figsize=(LF_SUMMARY_FIGURE_WIDTH, fig_height))

    row_positions = np.arange(len(agent_rows))[::-1]
    bar_height = 0.5
    theory_height = bar_height + 0.1
    max_right_limit = tsirelson_violation + 0.12

    for y_pos, row in zip(row_positions, agent_rows):
        left_exp = violation_offset
        left_ideal = violation_offset
        cumulative_centers = []
        cumulative_variance = 0.0

        for idx in range(len(LF_TERM_SPECS)):
            width_exp = abs(row["signed_values"][idx])
            width_ideal = abs(signed_ideal_values[idx])

            ax.barh(
                y_pos,
                width_exp,
                height=bar_height,
                left=left_exp,
                color=LF_TERM_COLORS[idx],
                edgecolor="none",
                zorder=2,
            )

            theory_left = left_ideal
            theory_right = left_ideal + width_ideal
            theory_ymin = y_pos - theory_height / 2.0
            theory_ymax = y_pos + theory_height / 2.0
            ax.hlines(
                [theory_ymin, theory_ymax],
                theory_left,
                theory_right,
                colors="red",
                linestyles="--",
                linewidth=1.5,
                zorder=3,
            )
            ax.vlines(
                [theory_left, theory_right],
                theory_ymin,
                theory_ymax,
                colors="red",
                linestyles="--",
                linewidth=1.5,
                zorder=3,
            )

            left_exp += width_exp
            left_ideal += width_ideal
            cumulative_variance += row["signed_errors"][idx] ** 2
            cumulative_centers.append(left_exp)

        s_summary = row["series"]["_s_summary"]
        final_sigma = s_summary["stderr"]
        final_violation = s_summary["value"]
        run_count = row["series"]["_run_count"]
        if run_count > 1:
            ax.errorbar(
                [cumulative_centers[-1]],
                [y_pos],
                xerr=[final_sigma / 2.0],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )

        final_text_x = cumulative_centers[-1] + final_sigma / 2.0 + 0.08
        ax.text(
            final_text_x,
            y_pos,
            f"S = {final_violation:.3f}\n± {final_sigma:.3f}",
            ha="left",
            va="center",
            fontsize=LF_SUMMARY_VALUE_LABEL_FONT_SIZE,
            zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.4},
        )
        max_right_limit = max(max_right_limit, final_text_x + 0.42)

        if show_epsilon_bounds:
            threshold_ymin = y_pos - (theory_height / 2.0 + 0.05)
            threshold_ymax = y_pos + (theory_height / 2.0 + 0.05)
            four_epsilon_markers = []
            four_epsilon = row["four_epsilon"]
            if four_epsilon is not None and np.isfinite(four_epsilon):
                four_epsilon_markers.append((float(four_epsilon), r"$4\epsilon$"))
            four_epsilon_max = row["four_epsilon_max"]
            if four_epsilon_max is not None and np.isfinite(four_epsilon_max):
                four_epsilon_markers.append((float(four_epsilon_max), r"$4\epsilon_{\max}$"))

            for marker_value, marker_label in four_epsilon_markers:
                ax.vlines(
                    marker_value,
                    threshold_ymin,
                    threshold_ymax,
                    colors="black",
                    linestyles="-",
                    linewidth=1.8,
                    zorder=6,
                )
                ax.text(
                    marker_value,
                    threshold_ymax + 0.04,
                    marker_label,
                    ha="center",
                    va="bottom",
                    fontsize=LF_SUMMARY_VALUE_LABEL_FONT_SIZE,
                    zorder=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
                )
                max_right_limit = max(max_right_limit, marker_value + 0.22)

    for threshold in [0.0, tsirelson_violation]:
        ax.axvline(
            x=threshold,
            color="red",
            linewidth=2.2,
            zorder=5,
        )

    ax.set_xlim(violation_offset, max_right_limit)
    ax.set_ylim(-0.7, len(agent_rows) - 0.3)
    ax.set_yticks(row_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y_pos, row in zip(row_positions, agent_rows):
        ax.text(
            LF_SUMMARY_AGENT_LABEL_X,
            y_pos,
            multiline_agent_display_label(row["agent_name"]),
            transform=label_transform,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=LF_SUMMARY_LABEL_FONT_SIZE,
            clip_on=False,
        )
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([violation_offset, 0.0, tsirelson_violation])
    set_thesis_xticklabels(ax, ["-2", "0", r"$2\sqrt{2}-2$"], fontsize=LF_LOWER_TICK_FONT_SIZE)
    ax.spines["bottom"].set_linewidth(1.5)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, (_, _, label) in zip(LF_TERM_COLORS, LF_TERM_SPECS)
    ]
    legend_handles.extend(
        [
            Line2D([], [], color="red", linestyle="--", linewidth=2, label="Ideal theoretical value"),
            Line2D([], [], color="black", linewidth=1.5, marker="|", markersize=10, label="Standard error of the mean"),
        ]
    )
    place_legend_above_axes(
        fig,
        ax,
        ncol=3,
        fontsize=LF_SUMMARY_LEGEND_FONT_SIZE,
        handles=legend_handles,
    )

    suffix = "_epsilon" if show_epsilon_bounds else ""
    plot_path = output_dir / f"hardware_agent_lf_violation_summary{suffix}.png"
    save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_accuracy_comparison(
    results,
    output_dir: Path,
    *,
    value_key: str,
    error_key: str,
    title: str,
    ylabel: str,
    ideal_value: float,
    ideal_label: str,
    show_legend: bool,
    y_max_override: Optional[float],
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    values = backend_values(results, value_key)
    errors = backend_values(results, error_key)
    x = np.arange(len(BACKEND_LABELS))
    max_reference = max(
        [ideal_value] + [value + error for value, error in zip(values, errors) if np.isfinite(value) and np.isfinite(error)]
    )
    if y_max_override is not None:
        y_max = float(y_max_override)
    elif np.isclose(ideal_value, 1.0):
        y_max = max(1.08, min(1.10, max_reference + 0.06))
    else:
        y_max = min(1.0, max_reference + 0.08)
    bar_width = 0.62
    target_width = 0.82

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        ecolor="#333333",
        width=bar_width,
        color=[ACCURACY_BACKEND_COLORS[label] for label in BACKEND_LABELS],
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    draw_accuracy_target_blocks(
        ax,
        x,
        [ideal_value] * len(x),
        width=target_width,
    )
    annotate_vertical_bars(
        ax,
        bars,
        values,
        errors=errors,
        upper_cap=y_max - 0.02,
        positive_offset=0.02,
        reference_values=[ideal_value] * len(values),
        fontsize=ANNOTATION_FONT_SIZE,
    )

    ax.set_xticks(x)
    set_thesis_xticklabels(ax, [result_axis_label(result_for_label(results, label)) for label in BACKEND_LABELS])
    ax.set_ylim(0.0, y_max)
    style_accuracy_axes(ax, ylabel)
    set_probability_axis_ticks(ax, y_max)
    if show_legend:
        place_legend_above_axes(
            fig,
            ax,
            ncol=1,
            fontsize=LEGEND_FONT_SIZE,
            handles=[Line2D([], [], color=ACCURACY_THEORY_LINE_COLOR, linestyle="--", linewidth=2.0, label=ideal_label)],
        )
    else:
        fig.tight_layout()

    plot_path = output_dir / filename
    save_plot(fig, plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_accuracy_metric(results, output_dir: Path, metric_key: str) -> Path:
    spec = ACCURACY_METRIC_SPECS[metric_key]
    return plot_accuracy_comparison(
        results,
        output_dir,
        value_key=spec["value_key"],
        error_key=spec["error_key"],
        title=spec["title"],
        ylabel=spec["ylabel"],
        ideal_value=spec["ideal_value"],
        ideal_label=spec["ideal_label"],
        show_legend=spec["show_legend"],
        y_max_override=spec["y_max"],
        filename=spec["filename"],
    )


def save_accuracy_metric_plot(results, output_dir: Path, metric_key: str) -> Path:
    plot_path = plot_accuracy_metric(results, output_dir, metric_key)
    save_plot_metadata(plot_path, build_accuracy_plot_metadata_for_metric(results, metric_key))
    return plot_path


def save_backend_lf_plot_group(results, output_dir: Path, backend_label: str, memory_inaccuracy_summary=None):
    plot_paths = plot_backend_lf_correlator_comparisons(
        results,
        output_dir,
        backend_label,
        memory_inaccuracy_summary,
    )
    for plot_path, agent_name in zip(plot_paths, LF_AGENT_NAMES):
        save_plot_metadata(plot_path, build_backend_lf_plot_metadata(results, backend_label, agent_name))
    return plot_paths


def plot_guessing_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_metric(results, output_dir, "guessing_accuracy")


def plot_reflex_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_metric(results, output_dir, "reflex_accuracy")


def plot_always_large_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_metric(results, output_dir, "always_large_accuracy")


def plot_reflex_sa_m_accuracy(results, output_dir: Path) -> Path:
    return plot_accuracy_metric(results, output_dir, "reflex_sa_m_accuracy")


# -----------------------------------------------------------------------------
# Printed summaries and script entry point

def print_accuracy_metric_summary(results, metric_key: str):
    spec = ACCURACY_METRIC_SPECS[metric_key]
    value_key = spec["value_key"]
    error_key = spec["error_key"]
    shots_key = spec["shots_key"]
    print(spec["summary_title"])
    for result in results:
        print(
            f"  {result_display_label(result)}: {result[value_key]:.4f} "
            f"+/- {result[error_key]:.4f} "
            f"(n={result[shots_key]})"
        )


def print_payoff_summary(results):
    print("\nExpected payoff comparison:")
    for result in results:
        print(
            f"  {result_display_label(result)}: "
            f"Betting Agent={result['observed_payoff']:.4f}, "
            f"Always-3/4={result['always_large_observed_payoff']:.4f}"
        )
    print(f"  Random agent (theory): {theory_payoff_for_policy('random'):.4f}")
    print(f"  Opposite agent (theory): {theory_payoff_for_policy('opposite'):.4f}")
    print(f"  Always-1/4 agent (theory): {theory_payoff_for_policy('always_small'):.4f}")
    print(f"  Betting Agent (theory): {theory_payoff_for_policy('betting'):.4f}")
    print(f"  Always-3/4 agent (theory): {theory_payoff_for_policy('always_large'):.4f}")


def print_guessing_summary(results):
    print_accuracy_metric_summary(results, "guessing_accuracy")


def print_reflex_summary(results):
    print_accuracy_metric_summary(results, "reflex_accuracy")


def print_reflex_sa_m_summary(results):
    print_accuracy_metric_summary(results, "reflex_sa_m_accuracy")


def print_always_large_summary(results):
    print_accuracy_metric_summary(results, "always_large_accuracy")


def print_selection_summary(results):
    print("\nSelected evaluation inputs:")
    for result in results:
        print(
            f"  {result_display_label(result)}: {result['selection_mode']} "
            f"({result['run_count']} run{'s' if result['run_count'] != 1 else ''})"
        )
        for run_name in result["run_names"]:
            print(f"    - {run_name}")


def evaluation_data_dirs(data_source: str) -> dict[str, Path]:
    if data_source == "paperdata":
        return {
            "Noiseless": PAPERDATA_DIR_NOISELESS,
            "Fake hardware": PAPERDATA_DIR_FAKE,
            "Real hardware": PAPERDATA_DIR_REAL,
        }
    if data_source == "latest-runs":
        return {
            "Noiseless": DATA_DIR_NOISELESS,
            "Fake hardware": DATA_DIR_FAKE,
            "Real hardware": DATA_DIR_REAL,
        }
    raise ValueError(f"Unknown data source: {data_source}")


def evaluation_last_n(args) -> dict[str, int]:
    if args.last_n is None:
        return dict(EVALUATION_LAST_N)
    return {label: args.last_n for label in BACKEND_LABELS}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create EWFS evaluation plots from saved experiment runs."
    )
    parser.add_argument(
        "--data-source",
        choices=["paperdata", "latest-runs"],
        default=EVALUATION_DATA_SOURCE,
        help=(
            "Use frozen thesis data from data/paperdata, or use the latest N "
            "runs from the normal data/data_* folders."
        ),
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Number of runs per backend when no explicit run folder is selected. Defaults to EVALUATION_LAST_N.",
    )
    parser.add_argument("--noiseless-run", type=str, default=None, help="Run folder name inside the selected noiseless data source.")
    parser.add_argument("--fake-run", type=str, default=None, help="Run folder name inside the selected fake-hardware data source.")
    parser.add_argument("--real-run", type=str, default=None, help="Run folder name inside the selected real-hardware data source.")
    args = parser.parse_args(argv)
    if args.last_n is not None and args.last_n <= 0:
        parser.error("--last-n must be a positive integer.")

    data_dirs = evaluation_data_dirs(args.data_source)
    last_n_by_label = evaluation_last_n(args)

    results = [
        load_backend_result(
            "Noiseless",
            data_dirs["Noiseless"],
            "noiseless_simulation.json",
            run_name=args.noiseless_run,
            last_n=last_n_by_label["Noiseless"],
            data_source=args.data_source,
        ),
        load_backend_result(
            "Fake hardware",
            data_dirs["Fake hardware"],
            "fake_hardware_noise_sim.json",
            run_name=args.fake_run,
            last_n=last_n_by_label["Fake hardware"],
            data_source=args.data_source,
        ),
        load_backend_result(
            "Real hardware",
            data_dirs["Real hardware"],
            "real_hardware_run.json",
            run_name=args.real_run,
            last_n=last_n_by_label["Real hardware"],
            data_source=args.data_source,
        ),
    ]
    memory_inaccuracy_results = []
    accuracy_specs = [
        ("Noiseless", data_dirs["Noiseless"], "noiseless_simulation.json", ACCURACY_TEST_RESULT_FILENAMES["Noiseless"], args.noiseless_run),
        ("Fake hardware", data_dirs["Fake hardware"], "fake_hardware_noise_sim.json", ACCURACY_TEST_RESULT_FILENAMES["Fake hardware"], args.fake_run),
        ("Real hardware", data_dirs["Real hardware"], "real_hardware_run.json", ACCURACY_TEST_RESULT_FILENAMES["Real hardware"], args.real_run),
    ]
    for label, data_dir, main_result_filename, accuracy_result_filename, run_name in accuracy_specs:
        main_result = result_for_label(results, label)
        try:
            memory_inaccuracy_results.append(
                load_accuracy_test_backend_result(
                    label,
                    data_dir,
                    main_result_filename,
                    accuracy_result_filename,
                    run_name=run_name,
                    last_n=last_n_by_label[label],
                    data_source=args.data_source,
                )
            )
        except FileNotFoundError as exc:
            memory_inaccuracy_results.append(missing_accuracy_test_result(label, main_result, exc))

    output_dir = build_output_dir()
    accuracy_dir = output_dir / "accuracy"
    reflex_agreement_dir = accuracy_dir / "reflex_agreement"
    comparison_dir = output_dir / "comparison"
    memory_initialization_dir = output_dir / "memory_initialization"
    correlators_dir = output_dir / "correlators_and_lf_values"
    correlators_noiseless_dir = correlators_dir / "noiseless"
    correlators_fake_dir = correlators_dir / "fake_hardware"
    correlators_real_dir = correlators_dir / "real_hardware"
    correlators_comparison_dir = correlators_dir / "comparison"
    memory_inaccuracy_summary = build_memory_inaccuracy_summary(memory_inaccuracy_results)
    memory_inaccuracy_folder_path = memory_initialization_dir / "memory_inaccuracy.json"
    tracking_epsilon_max_path = memory_initialization_dir / "tracking_epsilon_max_values.json"
    payoff_value_summary_path = comparison_dir / "payoff_comparison_values.json"
    tracking_epsilon_max_summary = build_tracking_epsilon_max_summary(results)
    payoff_value_summary = build_payoff_value_summary(results)
    save_json(memory_inaccuracy_folder_path, memory_inaccuracy_summary)
    save_json(tracking_epsilon_max_path, tracking_epsilon_max_summary)
    save_json(payoff_value_summary_path, payoff_value_summary)

    born_rule_accuracy_plot_path = plot_born_rule_accuracy(results, accuracy_dir)
    save_plot_metadata(
        born_rule_accuracy_plot_path,
        build_born_rule_plot_metadata(results),
    )
    always_large_vs_betting_plot_path = plot_always_large_vs_betting_payoff_comparison(results, comparison_dir)
    save_plot_metadata(
        always_large_vs_betting_plot_path,
        build_payoff_comparison_metadata(results),
    )
    lf_correlator_plot_paths = []
    noiseless_lf_plot_paths = save_backend_lf_plot_group(
        results,
        correlators_noiseless_dir,
        "Noiseless",
        memory_inaccuracy_summary,
    )
    lf_correlator_plot_paths.extend(noiseless_lf_plot_paths)
    real_lf_plot_paths = save_backend_lf_plot_group(
        results,
        correlators_real_dir,
        "Real hardware",
        memory_inaccuracy_summary,
    )
    lf_correlator_plot_paths.extend(real_lf_plot_paths)
    fake_lf_plot_paths = save_backend_lf_plot_group(
        results,
        correlators_fake_dir,
        "Fake hardware",
        memory_inaccuracy_summary,
    )
    lf_correlator_plot_paths.extend(fake_lf_plot_paths)
    hardware_comparison_lf_plot_paths = plot_hardware_lf_comparison_per_agent(
        results,
        correlators_comparison_dir,
        memory_inaccuracy_summary,
    )
    for plot_path, agent_name in zip(hardware_comparison_lf_plot_paths, LF_AGENT_NAMES):
        save_plot_metadata(plot_path, build_hardware_lf_comparison_metadata(results, agent_name))
    hardware_agent_summary_plot_path = plot_hardware_lf_agent_summary(
        results,
        correlators_comparison_dir,
        memory_inaccuracy_summary,
        show_epsilon_bounds=False,
    )
    save_plot_metadata(
        hardware_agent_summary_plot_path,
        build_hardware_lf_agent_summary_metadata(
            results,
            memory_inaccuracy_summary,
            show_epsilon_bounds=False,
        ),
    )
    hardware_agent_summary_epsilon_plot_path = plot_hardware_lf_agent_summary(
        results,
        correlators_comparison_dir,
        memory_inaccuracy_summary,
        show_epsilon_bounds=True,
    )
    save_plot_metadata(
        hardware_agent_summary_epsilon_plot_path,
        build_hardware_lf_agent_summary_metadata(
            results,
            memory_inaccuracy_summary,
            show_epsilon_bounds=True,
        ),
    )
    guessing_plot_path = save_accuracy_metric_plot(results, accuracy_dir, "guessing_accuracy")
    reflex_plot_path = save_accuracy_metric_plot(results, accuracy_dir, "reflex_accuracy")
    reflex_sa_m_plot_path = save_accuracy_metric_plot(results, reflex_agreement_dir, "reflex_sa_m_accuracy")
    always_large_accuracy_plot_path = save_accuracy_metric_plot(results, accuracy_dir, "always_large_accuracy")
    combined_memory_epsilon_plot_path = plot_combined_memory_epsilon(
        memory_inaccuracy_summary,
        memory_initialization_dir,
    )
    save_plot_metadata(
        combined_memory_epsilon_plot_path,
        build_memory_epsilon_plot_metadata(memory_inaccuracy_results),
    )
    print_selection_summary(results)
    print_payoff_summary(results)
    print_guessing_summary(results)
    print_reflex_summary(results)
    print_reflex_sa_m_summary(results)
    print_always_large_summary(results)
    print(f"Saved Betting Agent accuracy plot to: {born_rule_accuracy_plot_path} (PDF: {pdf_plot_path(born_rule_accuracy_plot_path)})")
    print(f"Saved betting-vs-always-3/4 payoff comparison plot to: {always_large_vs_betting_plot_path} (PDF: {pdf_plot_path(always_large_vs_betting_plot_path)})")
    for plot_path in lf_correlator_plot_paths:
        print(f"Saved backend LF correlator plot to: {plot_path} (PDF: {pdf_plot_path(plot_path)})")
    for plot_path in hardware_comparison_lf_plot_paths:
        print(f"Saved combined hardware LF correlator plot to: {plot_path} (PDF: {pdf_plot_path(plot_path)})")
    print(f"Saved hardware LF agent summary plot to: {hardware_agent_summary_plot_path} (PDF: {pdf_plot_path(hardware_agent_summary_plot_path)})")
    print(f"Saved hardware LF agent summary epsilon plot to: {hardware_agent_summary_epsilon_plot_path} (PDF: {pdf_plot_path(hardware_agent_summary_epsilon_plot_path)})")
    print(f"Saved guessing accuracy plot to: {guessing_plot_path} (PDF: {pdf_plot_path(guessing_plot_path)})")
    print(f"Saved reflex accuracy plot to: {reflex_plot_path} (PDF: {pdf_plot_path(reflex_plot_path)})")
    print(f"Saved reflex S_a/M agreement accuracy plot to: {reflex_sa_m_plot_path} (PDF: {pdf_plot_path(reflex_sa_m_plot_path)})")
    print(f"Saved always-3/4 accuracy plot to: {always_large_accuracy_plot_path} (PDF: {pdf_plot_path(always_large_accuracy_plot_path)})")
    print(f"Saved memory inaccuracy summary to: {memory_inaccuracy_folder_path}")
    print(f"Saved tracking epsilon-max values to: {tracking_epsilon_max_path}")
    print(f"Saved combined memory epsilon plot to: {combined_memory_epsilon_plot_path} (PDF: {pdf_plot_path(combined_memory_epsilon_plot_path)})")
    for result in memory_inaccuracy_results:
        if not result.get("available", True):
            print(f"Memory inaccuracy unavailable for {result['display_label']}: {result['error']}")


def evaluate_with_settings(*, data_source=EVALUATION_DATA_SOURCE, last_n=None):
    """Apply the simple front-door settings and create the evaluation plots."""
    global EVALUATION_DATA_SOURCE, EVALUATION_LAST_N

    EVALUATION_DATA_SOURCE = data_source
    if last_n is not None:
        EVALUATION_LAST_N = {
            "Noiseless": last_n,
            "Fake hardware": last_n,
            "Real hardware": last_n,
        }

    main(argv=[])


if __name__ == "__main__":
    main()
