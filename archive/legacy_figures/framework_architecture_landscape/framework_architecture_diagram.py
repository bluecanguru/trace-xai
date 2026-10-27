from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DARK = "#4F6A59"
MID = "#86AD96"
LIGHT = "#E8F0EB"
PALE = "#F5F7F5"
TEXT = "#1F2923"
GREY = "#65716A"
WHITE = "#FFFFFF"
ARROW = "#587264"
OPTIONAL_FILL = "#F1F4F2"
OPTIONAL_HEADER = "#71877A"
OPTIONAL_TEXT = "#526058"


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    *,
    optional=False,
    title_fontfamily="DejaVu Sans",
    body_fontsize=9.0,
    body_linespacing=1.42,
):
    """Draw one labelled architecture box on the diagram."""
    linestyle = (0, (5, 3)) if optional else "solid"
    fill_color = OPTIONAL_FILL if optional else PALE
    header_color = OPTIONAL_HEADER if optional else DARK
    body_color = OPTIONAL_TEXT if optional else TEXT
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.45,
        edgecolor=DARK,
        facecolor=fill_color,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(outer)

    header_h = min(0.052, h * 0.28)
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0,
        facecolor=header_color,
        zorder=3,
    )
    ax.add_patch(header)
    ax.add_patch(
        plt.Rectangle(
            (x, y + h - header_h),
            w,
            header_h / 2,
            linewidth=0,
            facecolor=header_color,
            zorder=3,
        )
    )

    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=10.3,
        fontweight="bold",
        fontfamily=title_fontfamily,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + (h - header_h) / 2,
        body,
        ha="center",
        va="center",
        color=body_color,
        fontsize=body_fontsize,
        linespacing=body_linespacing,
        zorder=4,
    )


def add_arrow(
    ax,
    start,
    end,
    *,
    dashed=False,
    shrink_a=2,
    shrink_b=2,
):
    """Draw one solid or dashed connector on the diagram."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color=ARROW,
            linestyle=(0, (4, 3)) if dashed else "solid",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            connectionstyle="arc3",
            zorder=6,
        )
    )


def add_split_connector(ax, *, source_x, source_y, rail_y, target_xs, target_y):
    """Draw one orthogonal split followed by independent vertical arrows."""
    left, right = min(target_xs), max(target_xs)
    ax.plot(
        [source_x, source_x],
        [source_y, rail_y],
        color=ARROW,
        linewidth=1.7,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.plot(
        [left, right],
        [rail_y, rail_y],
        color=ARROW,
        linewidth=1.7,
        solid_capstyle="butt",
        zorder=5,
    )
    for target_x in target_xs:
        add_arrow(
            ax,
            (target_x, rail_y),
            (target_x, target_y),
            shrink_a=0,
        )


def add_rich_line(ax, x, y, parts, *, fontsize=8.6):
    """Centre one line containing normal prose and monospace code fragments."""
    text_areas = [
        TextArea(
            value,
            textprops={
                "fontsize": fontsize,
                "fontfamily": "DejaVu Sans Mono" if is_code else "DejaVu Sans",
                "color": TEXT,
            },
        )
        for value, is_code in parts
    ]
    packed_line = HPacker(children=text_areas, align="center", pad=0, sep=0)
    ax.add_artist(
        AnnotationBbox(
            packed_line,
            (x, y),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=5,
        )
    )


def add_step(ax, x, y, number, label):
    """Draw a numbered workflow-stage label."""
    ax.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=WHITE,
        bbox=dict(boxstyle="circle,pad=0.28", facecolor=MID, edgecolor=DARK),
        zorder=7,
    )
    ax.text(
        x + 0.025,
        y,
        label,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=GREY,
        zorder=7,
    )


def main():
    """Generate the editable framework architecture assets."""
    output_dir = Path(__file__).resolve().parents[2] / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_step(ax, 0.045, 0.955, 1, "Inputs")
    add_box(
        ax,
        0.075,
        0.805,
        0.85,
        0.12,
        "Configuration and Analytical Inputs",
        "Data: fitted model  •  aligned train/explanation matrices  •  labels  •  ordered features\n"
        "Configuration: scoring rule  •  local settings  •  optional retraining callback",
    )

    add_arrow(ax, (0.50, 0.805), (0.50, 0.760))
    add_step(ax, 0.045, 0.720, 2, "Orchestration")
    add_box(
        ax,
        0.235,
        0.635,
        0.53,
        0.125,
        "run_pipeline(config)",
        "Stores aligned inputs and runs each enabled stage\n"
        "Collects results, provenance, versions, and execution times",
        title_fontfamily="DejaVu Sans Mono",
    )

    add_step(ax, 0.045, 0.620, 3, "Analysis branches")
    branch_y = 0.315
    branch_h = 0.25
    positions = [(0.035, 0.285), (0.3575, 0.285), (0.68, 0.285)]
    titles = [
        "Global Explanations",
        "Local Explanations",
        "Stability (Optional)",
    ]
    bodies = [
        "Native importance\nPermutation importance\nFeature shuffling with the\n"
        "configured performance scorer\nSHAP importance\nSpearman $\\rho$ rank agreement",
        "Representative outcome cases\nTP, TN, FP, and FN\n"
        "Borderline: $p$ nearest 0.5\nclassification threshold\n"
        "Repeated LIME: 20 seeds per case\nRule and seed-stability summaries",
        "100 stratified bootstrap refits\nPermutation importance (log loss)\n"
        "Spearman $\\rho$ rank stability\nTop-three overlap\nFeature frequency",
    ]

    for (x, w), title, body in zip(positions, titles, bodies):
        add_box(
            ax,
            x,
            branch_y,
            w,
            branch_h,
            title,
            body,
            optional=title.startswith("Stability"),
            body_fontsize=8.4,
            body_linespacing=1.25,
        )

    branch_centres = (0.1775, 0.50125, 0.8225)
    add_split_connector(
        ax,
        source_x=0.50,
        source_y=0.635,
        rail_y=0.585,
        target_xs=branch_centres,
        target_y=branch_y + branch_h,
    )

    add_step(ax, 0.045, 0.24, 4, "Outputs")
    for x in branch_centres:
        add_arrow(ax, (x, branch_y), (x, 0.205))
    add_box(
        ax,
        0.075,
        0.055,
        0.85,
        0.15,
        "Reproducible Outputs",
        "",
    )
    add_rich_line(
        ax,
        0.50,
        0.119,
        [
            ("Model artefacts: ", False),
            ("model.pkl", True),
            (" and fitted ", False),
            (".joblib", True),
            (" objects  •  metrics and tables: ", False),
            (".csv", True),
            (" and ", False),
            (".json", True),
        ],
    )
    add_rich_line(
        ax,
        0.50,
        0.087,
        [
            ("Execution logs: ", False),
            ("execution_times.json", True),
            ("  •  provenance: configuration, metadata, and library versions", False),
        ],
    )

    plt.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"framework_architecture.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor=WHITE,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()