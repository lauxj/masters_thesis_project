"""
Simple IBM backend coupling-map plots without calibration data.

This is intentionally the one simple backend-connectivity plotting script.
Set BACKEND_NAME to either "ibm_torino" or "ibm_marrakesh", and it will:
1. load the matching SVG geometry so the device still looks like a grid,
2. draw the full coupling map in black,
3. highlight the hard-coded Betting Agent layout,
4. create an agent-only comparison plot for Reflex, Guessing, Betting, and Always 3/4.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeTorino

EWFS_ROOT = Path(__file__).resolve().parents[1]
if str(EWFS_ROOT) not in sys.path:
    sys.path.insert(0, str(EWFS_ROOT))

from circuits.agents import (
    build_circuit_always_large,
    build_circuit_betting,
    build_circuit_guessing,
    build_circuit_reflex,
)

if not hasattr(np, "alltrue"):
    np.alltrue = np.all


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "plots_backend_connectivity"

# Change this to "ibm_torino" or "ibm_marrakesh".
BACKEND_NAME = "ibm_marrakesh"

TORINO_SVG_GEOMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "IBM_coupling_map"
    / "ibm_torino_readout_error_cz_calibrations_2026-03-30T03_26_45Z.svg"
)
MARRAKESH_SVG_GEOMETRY_CANDIDATES = [
    PROJECT_ROOT
    / "data"
    / "IBM_coupling_map"
    / "ibm_marrakesh_readout_error_cz_calibrations_2026-04-07T23_15_31Z.svg",
    Path("/Users/joshua/Downloads/ibm_marrakesh_readout_error_cz_calibrations_2026-04-07T23_15_31Z.svg"),
]

AGENT_BUILDERS = {
    "Reflex Agent": build_circuit_reflex,
    "Guessing Agent": build_circuit_guessing,
    "Always 3/4 Agent": build_circuit_always_large,
    "Betting Agent": build_circuit_betting,
}

AGENT_DISPLAY_LABELS = {
    "Always 3/4 Agent": "Always-3/4 Agent",
}


def agent_display_label(agent_name: str) -> str:
    return AGENT_DISPLAY_LABELS.get(agent_name, agent_name)


def ordered_qubit_names(circuit) -> list[str]:
    names: list[str] = []
    for qreg in circuit.qregs:
        if len(qreg) == 1:
            names.append(qreg.name)
        else:
            for index in range(len(qreg)):
                names.append(f"{qreg.name}[{index}]")
    return names


def iter_operation_qubit_indices(circuit) -> Iterable[list[int]]:
    global_index = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}

    def walk(subcircuit, qubit_map: dict[object, object]) -> Iterable[list[int]]:
        for instruction in subcircuit.data:
            operation = instruction.operation
            mapped_qubits = [qubit_map[qubit] for qubit in instruction.qubits]
            blocks = getattr(operation, "blocks", ())

            if blocks:
                for block in blocks:
                    if len(block.qubits) != len(mapped_qubits):
                        raise ValueError(f"Unsupported control-flow qubit mapping in {operation.name}.")
                    block_map = {
                        block_qubit: mapped_qubits[index]
                        for index, block_qubit in enumerate(block.qubits)
                    }
                    yield from walk(block, block_map)
                continue

            if operation.name in {"barrier", "measure"}:
                continue

            yield [global_index[qubit] for qubit in mapped_qubits]

    yield from walk(circuit, {qubit: qubit for qubit in circuit.qubits})


def summarize_circuit(circuit) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    activity: Counter[int] = Counter()

    for indices in iter_operation_qubit_indices(circuit):
        for index in indices:
            activity[index] += 1

        if len(indices) == 2:
            edge_counts[tuple(sorted(indices))] += 1
        elif len(indices) > 2:
            raise ValueError(
                "This plotter expects only one- and two-qubit operations after "
                f"expanding control-flow blocks, but found an operation on {len(indices)} qubits."
            )

    return dict(edge_counts), dict(activity)

ROLE_COLORS = {
    "core": "#2ca02c",
    "action": "#d62828",
    "choice": "#f4d35e",
}

BACKEND_SPECS = {
    "ibm_torino": {
        "backend_builder": FakeTorino,
        "svg_path": TORINO_SVG_GEOMETRY_PATH,
        "expected_qubits": 133,
        "output_prefix": OUTPUT_DIR / "ibm_torino_betting_layout_simple",
        "agent_output_prefix": OUTPUT_DIR / "ibm_torino_agent_connectivity_simple",
        "figure_size": (30, 22),
        "agent_figure_size": (25, 6.4),
        "node_size": 3600,
        "agent_node_size": 2400,
        "agent_backdrop_node_size": 360,
        "edge_width": 9,
        "agent_edge_width": 7,
        "agent_backdrop_edge_width": 2.2,
        "label_size": 32,
        "agent_label_size": 16,
        "title_size": 38,
        "agent_title_size": 26,
        "title": (
            "ibm_torino Coupling Map (March 30, 2026)\n"
            "Betting Agent layout without calibration data"
        ),
        "agent_title": (
            "ibm_torino agent connectivity (March 30, 2026)\n"
            "Reflex, Guessing, Always-3/4, and Betting agent structure only"
        ),
        "betting_layout": [68, 67, 66, 74, 65, 86, 18, 131],
        "agent_layouts": {
            "Reflex Agent": [54, 61, 62, 63, 14, 129],
            "Guessing Agent": [54, 61, 62, 60, 63, 14, 129],
            "Betting Agent": [68, 67, 66, 74, 65, 86, 18, 131],
            "Always 3/4 Agent": [68, 67, 66, 74, 65, 86, 18, 131],
        },
    },
    "ibm_marrakesh": {
        "backend_builder": FakeMarrakesh,
        "svg_path": None,
        "expected_qubits": 156,
        "output_prefix": OUTPUT_DIR / "ibm_marrakesh_betting_layout_simple",
        "agent_output_prefix": OUTPUT_DIR / "ibm_marrakesh_agent_connectivity_simple",
        "figure_size": (26, 18),
        "agent_figure_size": (26, 6.4),
        "node_size": 2050,
        "agent_node_size": 1900,
        "agent_backdrop_node_size": 280,
        "edge_width": 5,
        "agent_edge_width": 5,
        "agent_backdrop_edge_width": 2.2,
        "label_size": 18,
        "agent_label_size": 14,
        "title_size": 26,
        "agent_title_size": 22,
        "title": (
            "ibm_marrakesh Coupling Map (latest hard-coded real-hardware layout)\n"
            "Betting Agent layout without calibration data"
        ),
        "agent_title": (
            "ibm_marrakesh agent connectivity (latest hard-coded real-hardware layouts)\n"
            "Reflex, Guessing, Always-3/4, and Betting agent structure only"
        ),
        "betting_layout": [18, 11, 12, 10, 13, 9, 0, 155],
        "agent_layouts": {
            "Reflex Agent": [10, 11, 12, 13, 0, 155],
            "Guessing Agent": [10, 11, 12, 18, 13, 0, 155],
            "Betting Agent": [18, 11, 12, 10, 13, 9, 0, 155],
            "Always 3/4 Agent": [18, 11, 12, 10, 13, 9, 0, 155],
        },
    },
}


def resolve_marrakesh_svg_path() -> Path:
    """Return the first available Marrakesh SVG geometry file."""
    for path in MARRAKESH_SVG_GEOMETRY_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in MARRAKESH_SVG_GEOMETRY_CANDIDATES)
    raise FileNotFoundError(f"No Marrakesh SVG geometry file found. Checked:\n{candidates}")


def get_backend_positions(backend_name: str, graph: nx.Graph) -> dict[int, tuple[float, float]]:
    """Return backend qubit positions, falling back if SVG geometry is unavailable."""
    spec = BACKEND_SPECS[backend_name]
    svg_path = spec["svg_path"]
    if backend_name == "ibm_marrakesh":
        try:
            svg_path = resolve_marrakesh_svg_path()
        except FileNotFoundError:
            return nx.spring_layout(graph, seed=7)

    return load_svg_qubit_coordinates(svg_path, spec["expected_qubits"])


def load_svg_qubit_coordinates(svg_path: Path, expected_qubits: int) -> dict[int, tuple[float, float]]:
    """Recover fixed qubit coordinates from one IBM calibration SVG."""
    svg_text = svg_path.read_text()
    matches = re.findall(
        r'<g transform="translate\(([-\d.]+), ([-\d.]+)\)" '
        r'class="cursor-pointer transition-opacity" opacity="1"><circle r="8"',
        svg_text,
    )
    if len(matches) != expected_qubits:
        raise ValueError(f"Expected {expected_qubits} coordinates in {svg_path.name}, found {len(matches)}.")

    x_values = sorted({float(x) for x, _ in matches})
    y_values = sorted({float(y) for _, y in matches})
    x_step = median_grid_step(x_values)
    y_step = median_grid_step(y_values)
    target_step = 52.0

    return {
        index: (float(x) * target_step / x_step, -float(y) * target_step / y_step)
        for index, (x, y) in enumerate(matches)
    }


def median_grid_step(values: list[float]) -> float:
    """Return the median nonzero spacing between sorted grid coordinates."""
    steps = [
        right - left
        for left, right in zip(values, values[1:])
        if right - left > 1e-9
    ]
    if not steps:
        return 1.0
    return float(np.median(steps))


def build_backend_graph(backend_name: str) -> nx.Graph:
    """Build the undirected coupling graph for one backend."""
    backend = BACKEND_SPECS[backend_name]["backend_builder"]()
    graph = nx.Graph()
    graph.add_nodes_from(range(BACKEND_SPECS[backend_name]["expected_qubits"]))
    graph.add_edges_from({tuple(sorted(edge)) for edge in backend.coupling_map.get_edges()})
    return graph


def output_prefix_with_suffix(backend_name: str, suffix: str) -> Path:
    return BACKEND_SPECS[backend_name]["output_prefix"].with_suffix(f".{suffix}")


def agent_output_prefix_with_suffix(backend_name: str, suffix: str, stem_suffix: str = "") -> Path:
    prefix = BACKEND_SPECS[backend_name]["agent_output_prefix"]
    if stem_suffix:
        prefix = prefix.with_name(prefix.name + stem_suffix)
    return prefix.with_suffix(f".{suffix}")


def betting_edges_from_layout(layout: list[int]) -> list[tuple[int, int]]:
    """Return the highlighted Betting/Always-3/4 connectivity edges."""
    sb, sa, m1, m2, w0, w1, _, _ = layout
    return [
        tuple(sorted((sb, sa))),
        tuple(sorted((sa, m1))),
        tuple(sorted((m1, w0))),
        tuple(sorted((sa, m2))),
        tuple(sorted((m2, w1))),
    ]


def agent_role_for_name(qubit_name: str) -> str:
    if qubit_name in {"AC", "BC"}:
        return "choice"
    if qubit_name in {"SB", "SA"}:
        return "core"
    return "action"


def agent_plot_data(agent_name: str, layout: list[int]) -> dict[str, object]:
    circuit = AGENT_BUILDERS[agent_name]()
    logical_names = ordered_qubit_names(circuit)
    edge_counts, _ = summarize_circuit(circuit)
    layout_by_logical = dict(enumerate(layout))

    highlighted_edges = [
        tuple(sorted((layout_by_logical[left], layout_by_logical[right])))
        for left, right in sorted(edge_counts)
    ]
    core_nodes = {
        layout_by_logical[index]
        for index, name in enumerate(logical_names)
        if name in {"SB", "SA"}
    }
    green_edges = [
        edge
        for edge in highlighted_edges
        if set(edge) == core_nodes
    ]
    red_edges = [edge for edge in highlighted_edges if edge not in green_edges]
    labels = {
        physical: logical_names[index]
        for index, physical in enumerate(layout)
    }
    node_groups = {
        role: [
            physical
            for index, physical in enumerate(layout)
            if agent_role_for_name(logical_names[index]) == role
        ]
        for role in ROLE_COLORS
    }

    return {
        "labels": labels,
        "highlighted_edges": highlighted_edges,
        "green_edges": green_edges,
        "red_edges": red_edges,
        "logical_names": {
            physical: logical_names[index]
            for index, physical in enumerate(layout)
        },
        "node_groups": node_groups,
    }


def induced_edges(graph: nx.Graph, nodes: list[int]) -> list[tuple[int, int]]:
    return sorted(tuple(sorted(edge)) for edge in graph.subgraph(nodes).edges())


def draw_labels_with_choice_contrast(
    graph: nx.Graph,
    positions: dict[int, tuple[float, float]],
    labels: dict[int, str],
    choice_nodes: list[int],
    font_size: int,
    ax,
) -> None:
    """Draw white labels by default, but use black on yellow choice qubits."""
    choice_node_set = set(choice_nodes)
    standard_labels = {
        node: label
        for node, label in labels.items()
        if node not in choice_node_set
    }
    choice_labels = {
        node: label
        for node, label in labels.items()
        if node in choice_node_set
    }

    if standard_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=standard_labels,
            font_size=font_size,
            font_color="white",
            font_weight="bold",
            ax=ax,
        )
    if choice_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=choice_labels,
            font_size=font_size,
            font_color="black",
            font_weight="bold",
            ax=ax,
        )


def add_position_padding(
    positions: dict[int, tuple[float, float]],
    nodes: list[int],
    pad_fraction: float = 0.18,
    min_pad: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [positions[node][0] for node in nodes]
    ys = [positions[node][1] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    x_pad = max(x_span * pad_fraction, min_pad)
    y_pad = max(y_span * pad_fraction, min_pad)
    return (min_x - x_pad, max_x + x_pad), (min_y - y_pad, max_y + y_pad)


def compact_agent_positions(
    layout: list[int],
    plot_data: dict[str, object],
    positions: dict[int, tuple[float, float]],
    graph: nx.Graph,
) -> dict[int, tuple[float, float]]:
    """Keep the connected agent structure on-grid and move isolated choices beside it."""
    local_positions: dict[int, tuple[float, float]] = {}
    subgraph = graph.subgraph(layout)
    connected_nodes = [node for node in layout if subgraph.degree(node) > 0]
    isolated_nodes = [node for node in layout if subgraph.degree(node) == 0]
    horizontal_step = 52.0
    vertical_step = 52.0

    if connected_nodes:
        raw_xs = [positions[node][0] for node in connected_nodes]
        raw_ys = [positions[node][1] for node in connected_nodes]
        unique_xs = sorted(set(raw_xs))
        unique_ys = sorted(set(raw_ys), reverse=True)
        x_index = {x: index for index, x in enumerate(unique_xs)}
        y_index = {y: index for index, y in enumerate(unique_ys)}
        center_x = 0.5 * (len(unique_xs) - 1)
        center_y = 0.5 * (len(unique_ys) - 1)

        for node in connected_nodes:
            x, y = positions[node]
            local_positions[node] = (
                (x_index[x] - center_x) * horizontal_step,
                -(y_index[y] - center_y) * vertical_step,
            )

        xs = [local_positions[node][0] for node in connected_nodes]
        ys = [local_positions[node][1] for node in connected_nodes]
        min_x, max_x = min(xs), max(xs)
        center_y = 0.5 * (min(ys) + max(ys))
        x_span = max(max_x - min_x, horizontal_step)
        gap = max(0.82 * horizontal_step, 28.0)
    else:
        min_x = 0.0
        max_x = 0.0
        center_y = 0.0
        gap = 30.0

    logical_names = plot_data["logical_names"]
    left_candidates = [node for node in isolated_nodes if logical_names[node] == "AC"]
    right_candidates = [node for node in isolated_nodes if logical_names[node] == "BC"]
    remaining_isolates = [
        node
        for node in isolated_nodes
        if node not in left_candidates and node not in right_candidates
    ]

    for node in left_candidates:
        local_positions[node] = (min_x - gap, center_y)
    for node in right_candidates:
        local_positions[node] = (max_x + gap, center_y)

    for index, node in enumerate(remaining_isolates):
        offset = (index + 1) * 0.75 * gap
        side = -1 if index % 2 == 0 else 1
        anchor_x = min_x if side < 0 else max_x
        local_positions[node] = (anchor_x + side * offset, center_y)

    return local_positions


def plot_backend_betting_layout_simple(backend_name: str = BACKEND_NAME, save: bool = True):
    """Render a plain backend grid with the betting layout highlighted."""
    if backend_name not in BACKEND_SPECS:
        known = ", ".join(sorted(BACKEND_SPECS))
        raise ValueError(f"Unknown backend '{backend_name}'. Known backends: {known}")

    spec = BACKEND_SPECS[backend_name]
    graph = build_backend_graph(backend_name)
    positions = get_backend_positions(backend_name, graph)
    betting_layout = spec["betting_layout"]
    betting_edges = betting_edges_from_layout(betting_layout)
    sd_sc_edge = [tuple(sorted(betting_layout[:2]))]
    choice_qubits = betting_layout[-2:]
    sd_sc_qubits = betting_layout[:2]
    other_layout_qubits = [qubit for qubit in betting_layout if qubit not in choice_qubits + sd_sc_qubits]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=spec["figure_size"])
    title = fig.suptitle(spec["title"], fontsize=spec["title_size"], y=0.98)

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=list(graph.edges()),
        width=spec["edge_width"],
        edge_color="black",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=betting_edges,
        width=spec["edge_width"],
        edge_color="#d62828",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=sd_sc_edge,
        width=spec["edge_width"] + 2,
        edge_color="#2ca02c",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="black",
        node_size=spec["node_size"],
        edgecolors="black",
        linewidths=3,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=other_layout_qubits,
        node_color="#d62828",
        node_size=spec["node_size"],
        edgecolors="black",
        linewidths=3,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sd_sc_qubits,
        node_color="#2ca02c",
        node_size=spec["node_size"],
        edgecolors="black",
        linewidths=3,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=choice_qubits,
        node_color="#f4d35e",
        node_size=spec["node_size"],
        edgecolors="black",
        linewidths=3,
        ax=ax,
    )
    draw_labels_with_choice_contrast(
        graph,
        positions,
        labels={node: str(node) for node in graph.nodes()},
        choice_nodes=choice_qubits,
        font_size=spec["label_size"],
        ax=ax,
    )

    ax.axis("off")
    ax.set_aspect("equal")
    ax.margins(x=0.01, y=0.01)
    plt.tight_layout()

    if save:
        plt.savefig(output_prefix_with_suffix(backend_name, "png"), bbox_inches="tight", pad_inches=0.05)
        title.remove()
        plt.savefig(output_prefix_with_suffix(backend_name, "pdf"), bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        return None

    return fig


def _plot_backend_agent_connectivity_grid(
    backend_name: str,
    save: bool,
    rows: int,
    cols: int,
    figure_size: tuple[float, float],
    stem_suffix: str = "",
    box_aspect: float = 0.36,
    hspace: float | None = None,
):
    """Render each agent's connectivity using only its assigned qubits."""
    if backend_name not in BACKEND_SPECS:
        known = ", ".join(sorted(BACKEND_SPECS))
        raise ValueError(f"Unknown backend '{backend_name}'. Known backends: {known}")
    if rows * cols < len(AGENT_BUILDERS):
        raise ValueError(f"Grid {rows}x{cols} cannot fit {len(AGENT_BUILDERS)} agents.")

    spec = BACKEND_SPECS[backend_name]
    graph = build_backend_graph(backend_name)
    positions = get_backend_positions(backend_name, graph)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    axes = np.atleast_1d(axes).ravel()
    panel_data = []
    max_x_span = 0.0
    max_y_span = 0.0

    for agent_name in AGENT_BUILDERS:
        layout = spec["agent_layouts"][agent_name]
        plot_data = agent_plot_data(agent_name, layout)
        background_edges = induced_edges(graph, layout)
        local_positions = compact_agent_positions(layout, plot_data, positions, graph)
        x_limits, y_limits = add_position_padding(
            local_positions,
            layout,
            pad_fraction=0.20,
            min_pad=10.0,
        )
        panel_data.append(
            {
                "agent_name": agent_name,
                "layout": layout,
                "plot_data": plot_data,
                "background_edges": background_edges,
                "local_positions": local_positions,
                "x_limits": x_limits,
                "y_limits": y_limits,
            }
        )
        max_x_span = max(max_x_span, x_limits[1] - x_limits[0])
        max_y_span = max(max_y_span, y_limits[1] - y_limits[0])

    for ax, panel in zip(axes, panel_data):
        agent_name = panel["agent_name"]
        layout = panel["layout"]
        plot_data = panel["plot_data"]
        background_edges = panel["background_edges"]
        local_positions = panel["local_positions"]
        x_limits = panel["x_limits"]
        y_limits = panel["y_limits"]

        nx.draw_networkx_edges(
            graph,
            local_positions,
            edgelist=background_edges,
            width=max(spec["agent_edge_width"] - 1, 1),
            edge_color="black",
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            local_positions,
            edgelist=plot_data["red_edges"],
            width=spec["agent_edge_width"] + 1,
            edge_color="#d62828",
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            local_positions,
            edgelist=plot_data["green_edges"],
            width=spec["agent_edge_width"] + 1,
            edge_color="#2ca02c",
            ax=ax,
        )

        for role, color in ROLE_COLORS.items():
            nx.draw_networkx_nodes(
                graph,
                local_positions,
                nodelist=plot_data["node_groups"][role],
                node_color=color,
                node_size=spec["agent_node_size"],
                edgecolors="black",
                linewidths=3,
                ax=ax,
            )

        draw_labels_with_choice_contrast(
            graph,
            local_positions,
            labels=plot_data["labels"],
            choice_nodes=plot_data["node_groups"]["choice"],
            font_size=spec["agent_label_size"],
            ax=ax,
        )

        ax.set_title(agent_display_label(agent_name), fontsize=spec["agent_title_size"], y=0.96)
        x_center = 0.5 * (x_limits[0] + x_limits[1])
        y_center = 0.5 * (y_limits[0] + y_limits[1])
        ax.set_xlim(x_center - 0.5 * max_x_span, x_center + 0.5 * max_x_span)
        ax.set_ylim(y_center - 0.5 * max_y_span, y_center + 0.5 * max_y_span)
        ax.set_box_aspect(box_aspect)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(panel_data):]:
        ax.axis("off")

    plt.tight_layout(w_pad=0.05, pad=0.3)
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)

    if save:
        for suffix in ("png", "pdf"):
            plt.savefig(
                agent_output_prefix_with_suffix(backend_name, suffix, stem_suffix=stem_suffix),
                bbox_inches="tight",
                pad_inches=0.2,
            )
        plt.close(fig)
        return None

    return fig


def plot_backend_agent_connectivity_simple(backend_name: str = BACKEND_NAME, save: bool = True):
    """Render each agent's connectivity as a single-row comparison plot."""
    spec = BACKEND_SPECS[backend_name]
    return _plot_backend_agent_connectivity_grid(
        backend_name=backend_name,
        save=save,
        rows=1,
        cols=len(AGENT_BUILDERS),
        figure_size=spec["agent_figure_size"],
    )


def plot_backend_agent_connectivity_two_rows(backend_name: str = BACKEND_NAME, save: bool = True):
    """Render each agent's connectivity as a 2x2 comparison plot."""
    single_row_width, single_row_height = BACKEND_SPECS[backend_name]["agent_figure_size"]
    return _plot_backend_agent_connectivity_grid(
        backend_name=backend_name,
        save=save,
        rows=2,
        cols=2,
        figure_size=(single_row_width * 0.55, single_row_height * 1.05),
        stem_suffix="_two_rows",
        box_aspect=0.36,
        hspace=-0.20,
    )


if __name__ == "__main__":
    plot_backend_betting_layout_simple()
    plot_backend_agent_connectivity_simple()
    plot_backend_agent_connectivity_two_rows()
