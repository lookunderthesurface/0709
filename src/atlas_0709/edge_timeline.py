from __future__ import annotations

import html
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EdgeSpan:
    name: str
    start_s: float
    end_s: float

    @property
    def elapsed_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class EdgeRoundTimeline:
    round_index: int
    verify_blackbox: EdgeSpan
    forest_steps: list[EdgeSpan]
    post_prune_tree_steps: list[EdgeSpan]
    handoff: EdgeSpan | None
    forest_depth: int
    accepted_tokens: int
    generated_tokens: int
    handoff_mode: str
    verify_returned_before_forest_done: bool

    @property
    def forest_start_s(self) -> float:
        if not self.forest_steps:
            return self.verify_blackbox.start_s
        return self.forest_steps[0].start_s

    @property
    def forest_end_s(self) -> float:
        if not self.forest_steps:
            return self.verify_blackbox.start_s
        return self.forest_steps[-1].end_s

    @property
    def cloud_edge_overlap_s(self) -> float:
        verify = self.verify_blackbox
        return sum(
            max(0.0, min(verify.end_s, step.end_s) - max(verify.start_s, step.start_s))
            for step in self.forest_steps
        )

    @property
    def exposed_cloud_wait_s(self) -> float:
        return max(0.0, self.verify_blackbox.end_s - self.forest_end_s)

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": int(self.round_index),
            "verify_blackbox": self.verify_blackbox.to_dict(),
            "forest_steps": [span.to_dict() for span in self.forest_steps],
            "post_prune_tree_steps": [
                span.to_dict() for span in self.post_prune_tree_steps
            ],
            "handoff": None if self.handoff is None else self.handoff.to_dict(),
            "forest_depth": int(self.forest_depth),
            "accepted_tokens": int(self.accepted_tokens),
            "generated_tokens": int(self.generated_tokens),
            "handoff_mode": self.handoff_mode,
            "verify_returned_before_forest_done": bool(
                self.verify_returned_before_forest_done
            ),
            "cloud_edge_overlap_s": self.cloud_edge_overlap_s,
            "exposed_cloud_wait_s": self.exposed_cloud_wait_s,
        }


@dataclass
class EdgeBlackboxTimeline:
    edge_prefill: EdgeSpan
    cloud_prefill_blackbox: EdgeSpan
    initial_tree: EdgeSpan
    rounds: list[EdgeRoundTimeline] = field(default_factory=list)
    elapsed_s: float = 0.0
    generated_tokens: int = 0

    def summary(self) -> dict[str, object]:
        verify_times = [item.verify_blackbox.elapsed_s for item in self.rounds]
        overlaps = [item.cloud_edge_overlap_s for item in self.rounds]
        waits = [item.exposed_cloud_wait_s for item in self.rounds]
        total_verify = sum(verify_times)
        total_overlap = sum(overlaps)
        return {
            "measurement_view": "edge_observed_cloud_blackbox",
            "cloud_blackbox_definition": (
                "request serialization + network RTT + cloud queue + target verify "
                "+ target KV commit + response serialization"
            ),
            "cloud_internal_timings_used": False,
            "rounds": len(self.rounds),
            "generated_tokens": int(self.generated_tokens),
            "elapsed_s": float(self.elapsed_s),
            "tokens_per_second": (
                float(self.generated_tokens) / max(float(self.elapsed_s), 1e-12)
            ),
            "mean_verify_blackbox_s": (
                statistics.fmean(verify_times) if verify_times else 0.0
            ),
            "total_verify_blackbox_s": total_verify,
            "total_cloud_edge_overlap_s": total_overlap,
            "cloud_hidden_ratio": total_overlap / max(total_verify, 1e-12),
            "total_exposed_cloud_wait_s": sum(waits),
            "early_return_rounds": sum(
                int(item.verify_returned_before_forest_done) for item in self.rounds
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_prefill": self.edge_prefill.to_dict(),
            "cloud_prefill_blackbox": self.cloud_prefill_blackbox.to_dict(),
            "initial_tree": self.initial_tree.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "summary": self.summary(),
        }


def write_timeline_json(path: str | Path, timeline: EdgeBlackboxTimeline) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 12,
    fill: str = "#17202a",
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" letter-spacing="0" '
        'font-family="Inter, Segoe UI, Arial, sans-serif">'
        f"{html.escape(str(text))}</text>"
    )


def _svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    title: str,
    *,
    opacity: float = 1.0,
    stroke: str = "none",
    stroke_width: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0.8):.2f}" '
        f'height="{height:.2f}" rx="3" fill="{fill}" opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}">'
        f"<title>{html.escape(title)}</title></rect>"
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def write_timeline_svg(
    path: str | Path,
    timeline: EdgeBlackboxTimeline,
    *,
    width: int = 1600,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    width = max(int(width), 1100)
    left = 270
    right = 52
    top = 132
    row_height = 54
    bottom = 104
    row_count = 3 + len(timeline.rounds)
    height = top + row_count * row_height + bottom
    plot_width = width - left - right
    max_time = max(
        float(timeline.elapsed_s),
        timeline.edge_prefill.end_s,
        timeline.cloud_prefill_blackbox.end_s,
        timeline.initial_tree.end_s,
        *(
            max(
                item.verify_blackbox.end_s,
                item.forest_end_s,
                item.handoff.end_s if item.handoff is not None else 0.0,
            )
            for item in timeline.rounds
        ),
        1e-6,
    )

    def sx(value: float) -> float:
        return left + (float(value) / max_time) * plot_width

    summary = timeline.summary()
    rows: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(28, 34, "ATLAS 0709 Edge-observed timeline", size=21, weight=700),
        _svg_text(
            28,
            58,
            "Cloud is a black box: request send to response receive, including Target KV commit.",
            size=12,
            fill="#52606d",
        ),
        _svg_text(
            28,
            82,
            (
                f"total {timeline.elapsed_s * 1000.0:.1f} ms   "
                f"tokens {timeline.generated_tokens}   "
                f"throughput {float(summary['tokens_per_second']):.2f} tok/s   "
                f"mean cloud {float(summary['mean_verify_blackbox_s']) * 1000.0:.1f} ms   "
                f"hidden {float(summary['cloud_hidden_ratio']) * 100.0:.1f}%"
            ),
            size=12,
            fill="#2f3e46",
            weight=600,
        ),
    ]

    legend = [
        ("Cloud black box", "#d64545"),
        ("Edge prefill/tree", "#2878b5"),
        ("Edge forest step", "#2f9e69"),
        ("Post-prune tree step", "#3467a4"),
        ("Commit / prune", "#e09f3e"),
        ("Exposed cloud wait", "#7c8798"),
    ]
    legend_x = left
    for label, color in legend:
        rows.append(_svg_rect(legend_x, 96, 18, 11, color, label))
        rows.append(_svg_text(legend_x + 25, 106, label, size=11, fill="#465362"))
        legend_x += 172

    axis_y = top - 12
    rows.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_width}" '
        f'y2="{axis_y}" stroke="#9aa5b1" stroke-width="1"/>'
    )
    tick_count = 10
    for index in range(tick_count + 1):
        value = max_time * index / tick_count
        x = sx(value)
        rows.append(
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" '
            f'y2="{height - bottom + 12}" stroke="#e8ecef" stroke-width="1"/>'
        )
        rows.append(
            _svg_text(
                x,
                axis_y - 8,
                f"{value * 1000.0:.0f} ms",
                size=10,
                fill="#6b7785",
                anchor="middle",
            )
        )

    y = top

    def initial_row(label: str, span: EdgeSpan, color: str, detail: str) -> None:
        nonlocal y
        rows.append(_svg_text(28, y + 22, label, size=12, weight=600))
        rows.append(_svg_text(28, y + 39, detail, size=10, fill="#6b7785"))
        rows.append(
            _svg_rect(
                sx(span.start_s),
                y + 13,
                sx(span.end_s) - sx(span.start_s),
                17,
                color,
                f"{label}: {span.elapsed_s * 1000.0:.3f} ms",
            )
        )
        rows.append(
            _svg_text(
                min(sx(span.end_s) + 7, width - right),
                y + 26,
                f"{span.elapsed_s * 1000.0:.1f} ms",
                size=10,
                fill="#465362",
            )
        )
        y += row_height

    initial_row("Edge prompt prefill", timeline.edge_prefill, "#2878b5", "Drafter GPU")
    initial_row(
        "Cloud prompt black box",
        timeline.cloud_prefill_blackbox,
        "#d64545",
        "Edge POST /prefill RTT",
    )
    initial_row("Initial stage-1 tree", timeline.initial_tree, "#3a86a8", "k routes, depth d")

    for item in timeline.rounds:
        label = (
            f"Round {item.round_index:02d}  depth={item.forest_depth}  "
            f"accepted={item.accepted_tokens}  "
            f"forest={len(item.forest_steps)}  "
            f"post-prune tree={len(item.post_prune_tree_steps)}"
        )
        rows.append(_svg_text(28, y + 18, label, size=12, weight=600))
        rows.append(
            _svg_text(
                28,
                y + 37,
                item.handoff_mode,
                size=10,
                fill="#6b7785",
            )
        )
        verify = item.verify_blackbox
        rows.append(
            _svg_rect(
                sx(verify.start_s),
                y + 7,
                sx(verify.end_s) - sx(verify.start_s),
                14,
                "#d64545",
                (
                    f"Cloud black box round {item.round_index}: "
                    f"{verify.elapsed_s * 1000.0:.3f} ms"
                ),
            )
        )
        for step_index, step in enumerate(item.forest_steps):
            step_x = sx(step.start_s)
            step_width = sx(step.end_s) - sx(step.start_s)
            rows.append(
                _svg_rect(
                    step_x,
                    y + 25,
                    step_width,
                    11,
                    "#2f9e69",
                    (
                        f"Edge forest step {step_index + 1}: "
                        f"{step.elapsed_s * 1000.0:.3f} ms"
                    ),
                    stroke="#ffffff",
                    stroke_width=1.5,
                )
            )
        if item.exposed_cloud_wait_s > 0:
            rows.append(
                _svg_rect(
                    sx(item.forest_end_s),
                    y + 39,
                    sx(verify.end_s) - sx(item.forest_end_s),
                    8,
                    "#7c8798",
                    (
                        f"Exposed cloud wait: "
                        f"{item.exposed_cloud_wait_s * 1000.0:.3f} ms"
                    ),
                    opacity=0.85,
                )
            )
        if item.handoff is not None:
            handoff = item.handoff
            rows.append(
                _svg_rect(
                    sx(handoff.start_s),
                    y + 39,
                    sx(handoff.end_s) - sx(handoff.start_s),
                    8,
                    "#e09f3e",
                    (
                        f"Edge commit/prune/resume: "
                        f"{handoff.elapsed_s * 1000.0:.3f} ms"
                    ),
                    stroke="#ffffff",
                    stroke_width=1.0,
                )
            )
        for step_index, step in enumerate(item.post_prune_tree_steps):
            step_x = sx(step.start_s)
            step_width = sx(step.end_s) - sx(step.start_s)
            rows.append(
                _svg_rect(
                    step_x,
                    y + 39,
                    step_width,
                    8,
                    "#3467a4",
                    (
                        f"Post-prune build-tree step {step_index + 1}: "
                        f"{step.elapsed_s * 1000.0:.3f} ms"
                    ),
                    stroke="#ffffff",
                    stroke_width=1.5,
                )
            )
        rows.append(
            _svg_text(
                width - right + 5,
                y + 18,
                f"cloud {verify.elapsed_s * 1000.0:.1f} ms",
                size=9,
                fill="#7f1d1d",
            )
        )
        rows.append(
            f'<line x1="20" y1="{y + row_height - 2}" '
            f'x2="{width - 20}" y2="{y + row_height - 2}" '
            'stroke="#f0f2f4" stroke-width="1"/>'
        )
        y += row_height

    verify_values = [item.verify_blackbox.elapsed_s for item in timeline.rounds]
    overlap_values = [item.cloud_edge_overlap_s for item in timeline.rounds]
    wait_values = [item.exposed_cloud_wait_s for item in timeline.rounds]
    footer_y = height - 61
    rows.append(
        _svg_text(
            28,
            footer_y,
            (
                f"Mean verify black box {_mean(verify_values) * 1000.0:.2f} ms   "
                f"total overlap {sum(overlap_values) * 1000.0:.2f} ms   "
                f"exposed wait {sum(wait_values) * 1000.0:.2f} ms   "
                f"early-return rounds {int(summary['early_return_rounds'])}/{len(timeline.rounds)}"
            ),
            size=11,
            fill="#465362",
        )
    )
    rows.append(
        _svg_text(
            28,
            footer_y + 23,
            "Measured with time.perf_counter() in the Edge coordinator. No Cloud internal timer is used.",
            size=10,
            fill="#7b8794",
        )
    )
    rows.append("</svg>")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
