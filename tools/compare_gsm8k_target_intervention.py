from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from atlas_0709.eval_interventions import (
    aggregate_prediction_interventions,
    paired_target_quality,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "index" not in value:
                raise ValueError(f"invalid prediction row at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    predictions = path / "predictions.jsonl" if path.is_dir() else path
    if not predictions.is_file():
        raise FileNotFoundError(f"prediction JSONL does not exist: {predictions}")
    summary_path = predictions.parent / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"summary must contain a JSON object: {summary_path}")
        summary = value
    return _read_jsonl(predictions), summary, predictions


def _nested_policy(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("route_selection_policy")
    if isinstance(direct, str):
        return direct
    for key in ("metadata", "target_health", "generator_metadata"):
        found = _nested_policy(value.get(key))
        if found is not None:
            return found
    return None


def _run_policy(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str | None:
    # Runtime health is authoritative for remote Target runs.  The CLI setting
    # in summary.json is the fallback for legacy/in-process results.
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            found = _nested_policy(metadata.get("target_health"))
            if found is not None:
                return found
    settings = summary.get("settings")
    if isinstance(settings, Mapping):
        value = settings.get("route_selection_policy")
        if isinstance(value, str):
            return value
    return None


def _target_model(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    health = metadata.get("target_health")
    if not isinstance(health, Mapping):
        return None
    runtime = health.get("metadata")
    if not isinstance(runtime, Mapping):
        runtime = health
    value = runtime.get("model")
    return str(value) if value is not None else None


def _seed_signature(value: object, prefix: str = "") -> dict[str, Any]:
    signature: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return signature
    for raw_key, child in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        lowered = key.lower()
        if lowered == "seed" or lowered.endswith("_seed"):
            if isinstance(child, (str, int, float, bool)) or child is None:
                signature[path] = child
                continue
        if isinstance(child, Mapping):
            signature.update(_seed_signature(child, path))
    return signature


def _canonical_seed_values(signature: Mapping[str, Any]) -> list[tuple[str, Any]]:
    # Ignore container paths so a legacy top-level ``seed`` and a newer
    # ``reproducibility.seed`` remain comparable.  Role-specific suffixes stay
    # distinct (drafter_seed, target_seed, sample_seed, ...).
    values = [(path.rsplit(".", 1)[-1], value) for path, value in signature.items()]
    return sorted(values, key=lambda item: (item[0], repr(item[1])))


def _validate_alignment(
    best_rows: Sequence[Mapping[str, Any]],
    first_rows: Sequence[Mapping[str, Any]],
    best_summary: Mapping[str, Any],
    first_summary: Mapping[str, Any],
) -> dict[str, Any]:
    best_indices = [int(row["index"]) for row in best_rows]
    first_indices = [int(row["index"]) for row in first_rows]
    if len(set(best_indices)) != len(best_indices) or len(set(first_indices)) != len(first_indices):
        raise ValueError("prediction inputs contain duplicate indices")
    if set(best_indices) != set(first_indices):
        raise ValueError("target_best and first_route runs do not contain identical sample indices")

    best_by_index = {int(row["index"]): row for row in best_rows}
    first_by_index = {int(row["index"]): row for row in first_rows}
    row_seed_checked = 0
    row_seed_partially_unavailable = 0
    row_seed_unavailable = 0
    for index in sorted(best_by_index):
        best = best_by_index[index]
        first = first_by_index[index]
        for key in ("question", "gold", "model", "prompt_tokens"):
            if best.get(key) != first.get(key):
                raise ValueError(f"sample {index} differs in {key}")
        best_target_model = _target_model(best)
        first_target_model = _target_model(first)
        if (
            best_target_model is not None
            and first_target_model is not None
            and best_target_model != first_target_model
        ):
            raise ValueError(f"sample {index} uses different Target models")
        best_seed = _canonical_seed_values(_seed_signature(best))
        first_seed = _canonical_seed_values(_seed_signature(first))
        if best_seed and first_seed:
            row_seed_checked += 1
            if best_seed != first_seed:
                raise ValueError(
                    f"sample {index} has different seed metadata: "
                    f"target_best={best_seed}, first_route={first_seed}"
                )
        elif best_seed or first_seed:
            row_seed_partially_unavailable += 1
        else:
            row_seed_unavailable += 1

    comparable_setting_names = (
        "protocol",
        "num_fewshot",
        "max_new_tokens",
        "strict_marker",
        "k",
        "d",
        "drafter_do_sample",
        "drafter_temperature",
    )
    best_settings = best_summary.get("settings", {})
    first_settings = first_summary.get("settings", {})
    matched_settings: list[str] = []
    if isinstance(best_settings, Mapping) and isinstance(first_settings, Mapping):
        for name in comparable_setting_names:
            if name not in best_settings or name not in first_settings:
                continue
            if best_settings[name] != first_settings[name]:
                raise ValueError(
                    f"run summaries differ in paired setting {name}: "
                    f"target_best={best_settings[name]!r}, first_route={first_settings[name]!r}"
                )
            matched_settings.append(name)

    best_run_seed = _canonical_seed_values(_seed_signature(best_summary))
    first_run_seed = _canonical_seed_values(_seed_signature(first_summary))
    run_seed_status = "unavailable_legacy"
    if best_run_seed and first_run_seed:
        if best_run_seed != first_run_seed:
            raise ValueError(
                "run summaries have different seed metadata: "
                f"target_best={best_run_seed}, first_route={first_run_seed}"
            )
        run_seed_status = "matched"
    elif best_run_seed or first_run_seed:
        run_seed_status = "partially_unavailable"

    digest = hashlib.sha256(
        ",".join(str(index) for index in sorted(best_indices)).encode("utf-8")
    ).hexdigest()
    return {
        "sample_indices_matched": True,
        "sample_content_matched": True,
        "sample_count": len(best_indices),
        "sample_index_sha256": digest,
        "per_sample_seed_checked": row_seed_checked,
        "per_sample_seed_partially_unavailable": row_seed_partially_unavailable,
        "per_sample_seed_unavailable": row_seed_unavailable,
        "per_sample_seed_status": (
            "matched"
            if row_seed_checked == len(best_indices)
            else "partially_unavailable"
            if row_seed_checked or row_seed_partially_unavailable
            else "unavailable_legacy"
        ),
        "run_seed_status": run_seed_status,
        "run_seed_values": best_run_seed if run_seed_status == "matched" else None,
        "matched_run_settings": matched_settings,
        "forest_depth_intentionally_not_paired": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair target_best and first_route GSM8K predictions, then attribute "
            "helped/hurt answers to recorded Target route interventions."
        )
    )
    parser.add_argument(
        "--target-best",
        type=Path,
        required=True,
        help="target_best output directory or predictions.jsonl",
    )
    parser.add_argument(
        "--first-route",
        type=Path,
        required=True,
        help="first_route output directory or predictions.jsonl",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    best_rows, best_summary, best_path = _load_run(args.target_best)
    first_rows, first_summary, first_path = _load_run(args.first_route)
    best_policy = _run_policy(best_summary, best_rows)
    first_policy = _run_policy(first_summary, first_rows)
    if best_policy is not None and best_policy != "target_best":
        raise ValueError(f"--target-best input reports policy {best_policy!r}")
    if first_policy is not None and first_policy != "first_route":
        raise ValueError(f"--first-route input reports policy {first_policy!r}")

    result = {
        "comparison": "paired_target_best_vs_first_route",
        "target_best_predictions": str(best_path),
        "first_route_predictions": str(first_path),
        "policies": {
            "target_best": best_policy,
            "first_route": first_policy,
            "legacy_policy_metadata_missing": best_policy is None or first_policy is None,
        },
        "alignment": _validate_alignment(
            best_rows,
            first_rows,
            best_summary,
            first_summary,
        ),
        "target_best_intervention": aggregate_prediction_interventions(best_rows),
        "first_route_intervention": aggregate_prediction_interventions(first_rows),
        "paired_quality": paired_target_quality(best_rows, first_rows),
        "interpretation": {
            "helped": "target_best correct and first_route wrong",
            "hurt": "target_best wrong and first_route correct",
            "net_correct_gain": "helped - hurt",
            "causal_unit": (
                "paired final answer; round traces after the first differing token are not aligned"
            ),
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
