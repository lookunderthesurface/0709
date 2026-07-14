from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


INTERVENTION_SCHEMA_VERSION = 1


def _field(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _score_route_id(score: Mapping[str, Any]) -> int:
    return int(score["route_id"])


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _token_ids(score: Mapping[str, Any]) -> list[int]:
    values = score.get("token_ids")
    if not isinstance(values, (list, tuple)):
        return []
    return [int(value) for value in values]


def _draft_first_token_logprob(score: Mapping[str, Any]) -> float | None:
    """Read Drafter probabilities without confusing them with Target scores.

    New traces carry ``draft_token_logprobs``.  The scalar alias is accepted to
    make the evaluator tolerant of intermediate result formats.  In particular,
    ``first_token_logprob`` is deliberately *not* used: that field is the
    Target model's probability.
    """

    values = score.get("draft_token_logprobs")
    if isinstance(values, (list, tuple)) and values:
        return _finite_float(values[0])
    return _finite_float(score.get("draft_first_token_logprob"))


def _draft_path_logprob(score: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Return the path probability conditioned on the current round prefix.

    ``draft_logprob`` historically carried a cumulative beam score whose
    baseline could predate the current prefix after asynchronous forest reuse.
    New traces therefore provide ``draft_path_logprob`` and per-token values.
    The legacy score remains a last-resort fallback so old JSON is readable.
    """

    explicit = _finite_float(score.get("draft_path_logprob"))
    if explicit is not None:
        return explicit, "draft_path_logprob"
    values = score.get("draft_token_logprobs")
    if isinstance(values, (list, tuple)) and values:
        finite = [_finite_float(value) for value in values]
        if all(value is not None for value in finite):
            return math.fsum(float(value) for value in finite), "sum(draft_token_logprobs)"
    legacy = _finite_float(score.get("draft_logprob"))
    if legacy is not None:
        return legacy, "legacy_draft_logprob"
    return None, None


def _round_intervention(round_trace: object) -> dict[str, Any]:
    round_index = int(_field(round_trace, "round_index", -1))
    decision = str(_field(round_trace, "decision", "select"))
    selected_route_id_raw = _field(round_trace, "selected_route_id")
    scores_raw = _field(round_trace, "target_scores", [])
    scores = [dict(score) for score in scores_raw if isinstance(score, Mapping)]
    detail: dict[str, Any] = {
        "round_index": round_index,
        "decision": decision,
        "fallback": decision == "fallback_ar",
        "route_comparison_available": False,
    }
    if decision != "select" or selected_route_id_raw is None or not scores:
        return detail

    by_id = {_score_route_id(score): score for score in scores}
    selected_route_id = int(selected_route_id_raw)
    selected = by_id.get(selected_route_id)
    ranked_pairs = sorted(
        enumerate(scores),
        key=lambda pair: (
            -(
                path_score
                if (path_score := _draft_path_logprob(pair[1])[0]) is not None
                else float("-inf")
            ),
            int(pair[0]),
        ),
    )
    ranked = [score for _, score in ranked_pairs]
    if selected is None or not ranked:
        return detail

    draft_top = ranked[0]
    first_route = scores[0]
    draft_top_route_id = _score_route_id(draft_top)
    first_route_id = _score_route_id(first_route)
    draft_rank = next(
        index
        for index, score in enumerate(ranked, 1)
        if _score_route_id(score) == selected_route_id
    )
    selected_tokens = _token_ids(selected)
    draft_top_tokens = _token_ids(draft_top)
    first_route_tokens = _token_ids(first_route)
    selected_draft_total, selected_draft_source = _draft_path_logprob(selected)
    draft_top_total, draft_top_draft_source = _draft_path_logprob(draft_top)
    selected_draft_first = _draft_first_token_logprob(selected)
    draft_top_draft_first = _draft_first_token_logprob(draft_top)
    selected_target_first = _finite_float(selected.get("first_token_logprob"))
    draft_top_target_first = _finite_float(draft_top.get("first_token_logprob"))
    selected_target_total = _finite_float(selected.get("target_logprob"))
    draft_top_target_total = _finite_float(draft_top.get("target_logprob"))

    first_token_changed = bool(
        selected_tokens
        and draft_top_tokens
        and selected_tokens[0] != draft_top_tokens[0]
    )
    first_route_first_token_changed = bool(
        selected_tokens
        and first_route_tokens
        and selected_tokens[0] != first_route_tokens[0]
    )
    lower_draft_total = (
        None
        if selected_draft_total is None or draft_top_total is None
        else selected_draft_total < draft_top_total
    )
    lower_draft_first = (
        None
        if selected_draft_first is None or draft_top_draft_first is None
        else selected_draft_first < draft_top_draft_first
    )
    lower_target_first = (
        None
        if selected_target_first is None or draft_top_target_first is None
        else selected_target_first < draft_top_target_first
    )
    lower_target_total = (
        None
        if selected_target_total is None or draft_top_target_total is None
        else selected_target_total < draft_top_target_total
    )
    detail.update(
        {
            "route_comparison_available": True,
            "selected_route_id": selected_route_id,
            "draft_top_route_id": draft_top_route_id,
            "first_route_id": first_route_id,
            "selected_differs_from_draft_top": selected_route_id != draft_top_route_id,
            "selected_differs_from_first_route": selected_route_id != first_route_id,
            "draft_rank": draft_rank,
            "selected_first_token_id": selected_tokens[0] if selected_tokens else None,
            "draft_top_first_token_id": draft_top_tokens[0] if draft_top_tokens else None,
            "first_token_changed": first_token_changed,
            "first_route_first_token_changed": first_route_first_token_changed,
            "selected_draft_path_logprob": selected_draft_total,
            "draft_top_path_logprob": draft_top_total,
            "selected_draft_path_logprob_source": selected_draft_source,
            "draft_top_path_logprob_source": draft_top_draft_source,
            "current_prefix_draft_path_comparison": (
                selected_draft_source is not None
                and draft_top_draft_source is not None
                and selected_draft_source != "legacy_draft_logprob"
                and draft_top_draft_source != "legacy_draft_logprob"
            ),
            "draft_path_logprob_delta": (
                None
                if selected_draft_total is None or draft_top_total is None
                else selected_draft_total - draft_top_total
            ),
            "lower_total_draft_probability": lower_draft_total,
            "draft_first_token_comparison_available": lower_draft_first is not None,
            "selected_draft_first_token_logprob": selected_draft_first,
            "draft_top_first_token_logprob": draft_top_draft_first,
            "draft_first_token_logprob_delta": (
                None
                if selected_draft_first is None or draft_top_draft_first is None
                else selected_draft_first - draft_top_draft_first
            ),
            "lower_first_token_draft_probability": lower_draft_first,
            # These are useful diagnostics, but remain explicitly named as
            # Target probabilities so they cannot be mistaken for Drafter data.
            "lower_first_token_target_probability": lower_target_first,
            "target_first_token_logprob_delta": (
                None
                if selected_target_first is None or draft_top_target_first is None
                else selected_target_first - draft_top_target_first
            ),
            "lower_total_target_probability": lower_target_total,
            "target_total_logprob_delta": (
                None
                if selected_target_total is None or draft_top_target_total is None
                else selected_target_total - draft_top_target_total
            ),
        }
    )
    return detail


def summarize_generation_interventions(rounds: Sequence[object]) -> dict[str, Any]:
    """Summarize how often Target changed the Drafter's top complete route."""

    details = [_round_intervention(round_trace) for round_trace in rounds]
    comparable = [item for item in details if item["route_comparison_available"]]
    fallback_rounds = sum(bool(item["fallback"]) for item in details)
    intervention_rounds = sum(
        bool(item["selected_differs_from_draft_top"]) for item in comparable
    )
    first_route_intervention_rounds = sum(
        bool(item["selected_differs_from_first_route"]) for item in comparable
    )
    first_token_changed_rounds = sum(bool(item["first_token_changed"]) for item in comparable)
    first_route_first_token_changed_rounds = sum(
        bool(item["first_route_first_token_changed"]) for item in comparable
    )
    lower_total_draft_rounds = sum(
        item["lower_total_draft_probability"] is True for item in comparable
    )
    draft_first_comparable = [
        item for item in comparable if item["draft_first_token_comparison_available"]
    ]
    lower_first_draft_rounds = sum(
        item["lower_first_token_draft_probability"] is True
        for item in draft_first_comparable
    )
    rank_histogram = Counter(str(item["draft_rank"]) for item in comparable)
    non_top_ranks = [
        int(item["draft_rank"])
        for item in comparable
        if bool(item["selected_differs_from_draft_top"])
    ]
    return {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "definition": (
            "Causal Target action is selection away from first_route or fallback; "
            "probability displacement is selection away from the Drafter top path "
            "conditioned on the current round prefix"
        ),
        "target_action_definition": "selected_away_from_first_route_or_fallback",
        "route_intervention_definition": "selected_away_from_current_prefix_draft_top",
        "legacy_score_policy": (
            "legacy draft_logprob is used only when current-prefix path fields are absent"
        ),
        "round_count": len(details),
        "select_rounds": len(comparable),
        "fallback_rounds": fallback_rounds,
        "target_action_rounds": first_route_intervention_rounds + fallback_rounds,
        "target_action_rate": (
            (first_route_intervention_rounds + fallback_rounds) / len(details)
            if details
            else None
        ),
        "first_route_intervention_rounds": first_route_intervention_rounds,
        "first_route_intervention_rate": (
            first_route_intervention_rounds / len(comparable) if comparable else None
        ),
        "route_intervention_rounds": intervention_rounds,
        "route_intervention_rate": (
            intervention_rounds / len(comparable) if comparable else None
        ),
        "first_token_changed_rounds": first_token_changed_rounds,
        "first_route_first_token_changed_rounds": first_route_first_token_changed_rounds,
        "lower_total_draft_probability_rounds": lower_total_draft_rounds,
        "current_prefix_draft_path_comparison_rounds": sum(
            bool(item["current_prefix_draft_path_comparison"]) for item in comparable
        ),
        "draft_first_token_comparison_rounds": len(draft_first_comparable),
        "draft_first_token_comparison_coverage": (
            len(draft_first_comparable) / len(comparable) if comparable else None
        ),
        "lower_first_token_draft_probability_rounds": lower_first_draft_rounds,
        "draft_rank_histogram": dict(sorted(rank_histogram.items(), key=lambda item: int(item[0]))),
        "max_intervened_draft_rank": max(non_top_ranks) if non_top_ranks else None,
        "any_target_action": first_route_intervention_rounds + fallback_rounds > 0,
        "any_first_route_intervention": first_route_intervention_rounds > 0,
        "any_route_intervention": intervention_rounds > 0,
        "any_fallback": fallback_rounds > 0,
        "any_first_token_changed": first_token_changed_rounds > 0,
        "any_first_route_first_token_changed": first_route_first_token_changed_rounds > 0,
        "any_lower_total_draft_probability": lower_total_draft_rounds > 0,
        "any_lower_first_token_draft_probability": lower_first_draft_rounds > 0,
        "rounds": details,
    }


def prediction_intervention(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = row.get("target_intervention")
    if isinstance(direct, Mapping):
        return direct
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("target_intervention")
        if isinstance(nested, Mapping):
            return nested
    return None


def aggregate_prediction_interventions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [item for row in rows if (item := prediction_intervention(row)) is not None]

    def total(name: str) -> int:
        return sum(int(item.get(name, 0)) for item in available)

    select_rounds = total("select_rounds")
    draft_first_comparable = total("draft_first_token_comparison_rounds")
    rank_histogram: Counter[str] = Counter()
    for item in available:
        histogram = item.get("draft_rank_histogram", {})
        if isinstance(histogram, Mapping):
            rank_histogram.update({str(key): int(value) for key, value in histogram.items()})
    samples_with_intervention = sum(bool(item.get("any_route_intervention")) for item in available)
    samples_with_first_route_intervention = sum(
        bool(item.get("any_first_route_intervention")) for item in available
    )
    samples_with_target_action = sum(bool(item.get("any_target_action")) for item in available)
    return {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "target_action_definition": "selected_away_from_first_route_or_fallback",
        "route_intervention_definition": "selected_away_from_current_prefix_draft_top",
        "sample_count": len(rows),
        "samples_with_statistics": len(available),
        "sample_coverage": len(available) / len(rows) if rows else None,
        "samples_with_target_action": samples_with_target_action,
        "sample_target_action_rate": (
            samples_with_target_action / len(available) if available else None
        ),
        "samples_with_first_route_intervention": samples_with_first_route_intervention,
        "samples_with_route_intervention": samples_with_intervention,
        "sample_route_intervention_rate": (
            samples_with_intervention / len(available) if available else None
        ),
        "round_count": total("round_count"),
        "select_rounds": select_rounds,
        "fallback_rounds": total("fallback_rounds"),
        "target_action_rounds": total("target_action_rounds"),
        "target_action_rate": (
            total("target_action_rounds") / total("round_count")
            if total("round_count")
            else None
        ),
        "first_route_intervention_rounds": total("first_route_intervention_rounds"),
        "first_route_intervention_rate": (
            total("first_route_intervention_rounds") / select_rounds
            if select_rounds
            else None
        ),
        "route_intervention_rounds": total("route_intervention_rounds"),
        "route_intervention_rate": (
            total("route_intervention_rounds") / select_rounds if select_rounds else None
        ),
        "first_token_changed_rounds": total("first_token_changed_rounds"),
        "first_route_first_token_changed_rounds": total(
            "first_route_first_token_changed_rounds"
        ),
        "lower_total_draft_probability_rounds": total(
            "lower_total_draft_probability_rounds"
        ),
        "current_prefix_draft_path_comparison_rounds": total(
            "current_prefix_draft_path_comparison_rounds"
        ),
        "draft_first_token_comparison_rounds": draft_first_comparable,
        "draft_first_token_comparison_coverage": (
            draft_first_comparable / select_rounds if select_rounds else None
        ),
        "lower_first_token_draft_probability_rounds": total(
            "lower_first_token_draft_probability_rounds"
        ),
        "draft_rank_histogram": dict(
            sorted(rank_histogram.items(), key=lambda item: int(item[0]))
        ),
    }


def mcnemar_exact_p_value(helped: int, hurt: int) -> float | None:
    discordant = int(helped) + int(hurt)
    if discordant == 0:
        return None
    smaller = min(int(helped), int(hurt))
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(value + 1)
        - math.lgamma(discordant - value + 1)
        - discordant * math.log(2.0)
        for value in range(smaller + 1)
    ]
    max_log = max(log_terms)
    lower_tail = math.exp(max_log) * math.fsum(
        math.exp(value - max_log) for value in log_terms
    )
    return min(1.0, 2.0 * lower_tail)


def paired_target_quality(
    target_best_rows: Sequence[Mapping[str, Any]],
    first_route_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare paired GSM8K outcomes after callers validate sample alignment."""

    best_by_index = {int(row["index"]): row for row in target_best_rows}
    first_by_index = {int(row["index"]): row for row in first_route_rows}
    if len(best_by_index) != len(target_best_rows) or len(first_by_index) != len(first_route_rows):
        raise ValueError("prediction inputs contain duplicate sample indices")
    if set(best_by_index) != set(first_by_index):
        missing_best = sorted(set(first_by_index) - set(best_by_index))
        missing_first = sorted(set(best_by_index) - set(first_by_index))
        raise ValueError(
            "paired sample indices differ: "
            f"missing_target_best={missing_best[:10]}, missing_first_route={missing_first[:10]}"
        )

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index in sorted(best_by_index):
        best = best_by_index[index]
        first = first_by_index[index]
        for name in ("question", "gold", "model"):
            if best.get(name) != first.get(name):
                raise ValueError(f"sample {index} has mismatched {name}")
        pairs.append((best, first))

    helped = sum(bool(best.get("correct")) and not bool(first.get("correct")) for best, first in pairs)
    hurt = sum(not bool(best.get("correct")) and bool(first.get("correct")) for best, first in pairs)
    both_correct = sum(bool(best.get("correct")) and bool(first.get("correct")) for best, first in pairs)
    both_wrong = len(pairs) - helped - hurt - both_correct

    def impact_for(flag: str, *, require_available: bool = False) -> dict[str, Any]:
        qualified: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        stats_available = 0
        metric_available = 0
        for pair in pairs:
            stats = prediction_intervention(pair[0])
            if stats is None:
                continue
            stats_available += 1
            if require_available and int(stats.get("draft_first_token_comparison_rounds", 0)) <= 0:
                continue
            metric_available += 1
            if bool(stats.get(flag)):
                qualified.append(pair)
        local_helped = sum(
            bool(best.get("correct")) and not bool(first.get("correct"))
            for best, first in qualified
        )
        local_hurt = sum(
            not bool(best.get("correct")) and bool(first.get("correct"))
            for best, first in qualified
        )
        return {
            "samples_with_statistics": stats_available,
            "samples_with_comparable_metric": metric_available,
            "sample_count": len(qualified),
            "helped": local_helped,
            "hurt": local_hurt,
            "net_correct_gain": local_helped - local_hurt,
        }

    target_best_correct = both_correct + helped
    first_route_correct = both_correct + hurt
    return {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "pair_count": len(pairs),
        "target_best_correct": target_best_correct,
        "first_route_correct": first_route_correct,
        "target_best_accuracy": target_best_correct / len(pairs) if pairs else None,
        "first_route_accuracy": first_route_correct / len(pairs) if pairs else None,
        "accuracy_delta": (
            (target_best_correct - first_route_correct) / len(pairs) if pairs else None
        ),
        "helped": helped,
        "hurt": hurt,
        "net_correct_gain": helped - hurt,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "response_changed": sum(
            best.get("response") != first.get("response") for best, first in pairs
        ),
        "mcnemar_exact_two_sided_p": mcnemar_exact_p_value(helped, hurt),
        "impact_by_target_best_intervention": {
            "any_target_action": impact_for("any_target_action"),
            "selected_away_from_first_route": impact_for("any_first_route_intervention"),
            "route_selected_below_draft_top": impact_for("any_route_intervention"),
            "first_token_changed": impact_for("any_first_token_changed"),
            "first_route_first_token_changed": impact_for(
                "any_first_route_first_token_changed"
            ),
            "lower_total_draft_probability": impact_for(
                "any_lower_total_draft_probability"
            ),
            "lower_first_token_draft_probability": impact_for(
                "any_lower_first_token_draft_probability", require_available=True
            ),
            "fallback_ar": impact_for("any_fallback"),
        },
    }
