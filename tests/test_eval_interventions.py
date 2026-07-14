from __future__ import annotations

import math

from atlas_0709.eval_interventions import (
    aggregate_prediction_interventions,
    mcnemar_exact_p_value,
    paired_target_quality,
    summarize_generation_interventions,
)
from tools.compare_gsm8k_target_intervention import _validate_alignment


def _score(
    route_id: int,
    token_ids: list[int],
    *,
    draft: float,
    draft_tokens: list[float] | None,
    target: float,
    target_first: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "route_id": route_id,
        "token_ids": token_ids,
        "draft_logprob": draft,
        "target_logprob": target,
        "first_token_logprob": target_first,
    }
    if draft_tokens is not None:
        result["draft_token_logprobs"] = draft_tokens
    return result


def test_generation_summary_counts_target_route_intervention() -> None:
    scores = [
        _score(
            10,
            [100, 101],
            draft=-0.4,
            draft_tokens=[-0.1, -0.3],
            target=-1.2,
            target_first=-0.7,
        ),
        _score(
            20,
            [200, 201],
            draft=-0.9,
            draft_tokens=[-0.3, -0.6],
            target=-0.2,
            target_first=-0.1,
        ),
    ]
    summary = summarize_generation_interventions(
        [
            {
                "round_index": 0,
                "decision": "select",
                "selected_route_id": 20,
                "target_scores": scores,
            },
            {
                "round_index": 1,
                "decision": "fallback_ar",
                "selected_route_id": None,
                "target_scores": scores,
            },
        ]
    )

    assert summary["round_count"] == 2
    assert summary["select_rounds"] == 1
    assert summary["fallback_rounds"] == 1
    assert summary["target_action_rounds"] == 2
    assert summary["target_action_rate"] == 1.0
    assert summary["first_route_intervention_rate"] == 1.0
    assert summary["route_intervention_rounds"] == 1
    assert summary["first_token_changed_rounds"] == 1
    assert summary["lower_total_draft_probability_rounds"] == 1
    assert summary["lower_first_token_draft_probability_rounds"] == 1
    assert summary["draft_rank_histogram"] == {"2": 1}
    detail = summary["rounds"][0]
    assert detail["draft_rank"] == 2
    assert math.isclose(detail["draft_path_logprob_delta"], -0.5)
    assert math.isclose(detail["draft_first_token_logprob_delta"], -0.2)
    # Target probability is independently named and has the opposite sign in
    # this fixture; it must never be used as the Drafter first-token metric.
    assert detail["lower_first_token_target_probability"] is False


def test_legacy_scores_report_missing_draft_first_token_coverage() -> None:
    scores = [
        _score(1, [10], draft=-0.1, draft_tokens=None, target=-0.5, target_first=-0.5),
        _score(2, [20], draft=-0.2, draft_tokens=None, target=-0.1, target_first=-0.1),
    ]
    summary = summarize_generation_interventions(
        [
            {
                "round_index": 0,
                "decision": "select",
                "selected_route_id": 2,
                "target_scores": scores,
            }
        ]
    )

    assert summary["route_intervention_rounds"] == 1
    assert summary["draft_first_token_comparison_rounds"] == 0
    assert summary["draft_first_token_comparison_coverage"] == 0.0
    assert summary["rounds"][0]["lower_first_token_draft_probability"] is None


def test_first_route_intervention_is_separate_from_draft_top_rank() -> None:
    # After asynchronous promotion, route0 and the best current-prefix path can
    # be different.  Paired target_best-vs-first_route attribution must use
    # route0, while draft rank remains a separate probability diagnostic.
    scores = [
        _score(10, [10], draft=-1.0, draft_tokens=[-1.0], target=-1.0, target_first=-1.0),
        _score(20, [20], draft=-0.2, draft_tokens=[-0.2], target=-0.1, target_first=-0.1),
    ]
    summary = summarize_generation_interventions(
        [
            {
                "round_index": 0,
                "decision": "select",
                "selected_route_id": 20,
                "target_scores": scores,
            }
        ]
    )

    assert summary["route_intervention_rounds"] == 0
    assert summary["first_route_intervention_rounds"] == 1
    assert summary["target_action_rounds"] == 1
    assert summary["any_target_action"] is True


def test_prediction_aggregate_and_paired_quality_helped_hurt() -> None:
    intervention = summarize_generation_interventions(
        [
            {
                "round_index": 0,
                "decision": "select",
                "selected_route_id": 2,
                "target_scores": [
                    _score(
                        1,
                        [10],
                        draft=-0.1,
                        draft_tokens=[-0.1],
                        target=-0.5,
                        target_first=-0.5,
                    ),
                    _score(
                        2,
                        [20],
                        draft=-0.2,
                        draft_tokens=[-0.2],
                        target=-0.1,
                        target_first=-0.1,
                    ),
                ],
            }
        ]
    )
    best = [
        {
            "index": 0,
            "model": "draft",
            "question": "q0",
            "gold": "0",
            "correct": True,
            "response": "best0",
            "target_intervention": intervention,
        },
        {
            "index": 1,
            "model": "draft",
            "question": "q1",
            "gold": "1",
            "correct": False,
            "response": "best1",
            "target_intervention": intervention,
        },
        {
            "index": 2,
            "model": "draft",
            "question": "q2",
            "gold": "2",
            "correct": True,
            "response": "same",
            "target_intervention": intervention,
        },
    ]
    first = [
        {"index": 0, "model": "draft", "question": "q0", "gold": "0", "correct": False, "response": "first0"},
        {"index": 1, "model": "draft", "question": "q1", "gold": "1", "correct": True, "response": "first1"},
        {"index": 2, "model": "draft", "question": "q2", "gold": "2", "correct": True, "response": "same"},
    ]

    aggregate = aggregate_prediction_interventions(best)
    assert aggregate["samples_with_route_intervention"] == 3
    assert aggregate["sample_target_action_rate"] == 1.0
    assert aggregate["target_action_rate"] == 1.0
    assert aggregate["route_intervention_rounds"] == 3
    paired = paired_target_quality(best, first)
    assert paired["helped"] == 1
    assert paired["hurt"] == 1
    assert paired["net_correct_gain"] == 0
    assert paired["response_changed"] == 2
    impact = paired["impact_by_target_best_intervention"]["first_token_changed"]
    assert impact == {
        "samples_with_statistics": 3,
        "samples_with_comparable_metric": 3,
        "sample_count": 3,
        "helped": 1,
        "hurt": 1,
        "net_correct_gain": 0,
    }


def test_paired_quality_rejects_misaligned_samples() -> None:
    try:
        paired_target_quality(
            [{"index": 0, "model": "m", "question": "q", "gold": "1"}],
            [{"index": 1, "model": "m", "question": "q", "gold": "1"}],
        )
    except ValueError as exc:
        assert "indices differ" in str(exc)
    else:
        raise AssertionError("misaligned paired inputs should fail")


def test_mcnemar_exact_p_value() -> None:
    assert mcnemar_exact_p_value(0, 0) is None
    assert math.isclose(mcnemar_exact_p_value(5, 0), 0.0625)
    assert math.isclose(mcnemar_exact_p_value(1, 1), 1.0)
    full_gsm8k = mcnemar_exact_p_value(700, 619)
    assert full_gsm8k is not None and 0.0 <= full_gsm8k <= 1.0


def test_paired_alignment_requires_matching_prompt_token_hashes() -> None:
    base = {
        "index": 0,
        "backend": "atlas_serial",
        "model": "draft",
        "question": "q",
        "gold": "1",
        "prompt_tokens": 10,
    }
    try:
        _validate_alignment(
            [{**base, "prompt_token_sha256": "a"}],
            [{**base, "prompt_token_sha256": "b"}],
            {},
            {},
        )
    except ValueError as exc:
        assert "different prompt token IDs" in str(exc)
    else:
        raise AssertionError("different prompt token IDs should invalidate a pair")


def test_paired_alignment_requires_fallback_disabled() -> None:
    row = {
        "index": 0,
        "backend": "atlas_serial",
        "model": "draft",
        "question": "q",
        "gold": "1",
        "prompt_tokens": 10,
        "prompt_token_sha256": "same",
    }
    best_summary = {
        "settings": {"fallback_threshold": -0.5, "first_token_threshold": None}
    }
    first_summary = {
        "settings": {"fallback_threshold": None, "first_token_threshold": None}
    }
    try:
        _validate_alignment([row], [row], best_summary, first_summary)
    except ValueError as exc:
        assert "requires fallback disabled" in str(exc)
    else:
        raise AssertionError("enabled fallback should invalidate route-selection attribution")
