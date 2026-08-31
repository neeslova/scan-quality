"""Метки, исключённые из вердикта: считаются и видны, но решения не принимают.

Нужно это из-за метки, вырожденной на конкретном корпусе. На `Data iz tg`
такова `low_resolution`: медиана `line_height_px` у хороших и плохих страниц
одинакова — 12 пикселей там и там, — а с якорями от Tobacco (good 30 / bad 13)
каждая страница получала ровно 1.000. При правиле «хоть одна метка выше порога»
этого хватало, чтобы 188 страниц из 199 стали браком, а ROC-AUC упал до 0.505.
"""

from __future__ import annotations

from src.config import VerdictConfig
from src.pipeline import decide_verdict, deciding_scores, quality_score

CFG = VerdictConfig(tau_low=0.30, tau_high=0.60, tau_unreadable=0.40)
IGNORING = VerdictConfig(
    tau_low=0.30, tau_high=0.60, tau_unreadable=0.40, ignore=["low_resolution"]
)


def test_ignored_label_does_not_decide_the_verdict() -> None:
    scores = {"low_resolution": 1.0, "blur": 0.05}

    assert decide_verdict(scores, CFG) == "bad"
    assert decide_verdict(scores, IGNORING) == "good"


def test_ignored_label_stays_out_of_the_summary_score() -> None:
    """Иначе метка вернулась бы через сводный балл: по нему считается ROC-AUC."""
    scores = {"low_resolution": 1.0, "blur": 0.05}

    assert quality_score(scores, CFG) == 0.0
    assert quality_score(scores, IGNORING) == 0.95


def test_other_labels_still_decide() -> None:
    """Исключение адресное: остальные метки работают как прежде."""
    scores = {"low_resolution": 1.0, "shadow": 0.75}

    assert decide_verdict(scores, IGNORING) == "bad"


def test_ignoring_unreadable_disarms_its_own_threshold() -> None:
    """У `unreadable` отдельный порог, и он тоже обязан уважать исключение."""
    scores = {"unreadable": 0.9}
    ignoring = VerdictConfig(
        tau_low=0.30, tau_high=0.60, tau_unreadable=0.40, ignore=["unreadable"]
    )

    assert decide_verdict(scores, CFG) == "bad"
    assert decide_verdict(scores, ignoring) == "good"


def test_deciding_scores_keeps_everything_by_default() -> None:
    scores = {"blur": 0.1, "noise": 0.2}

    assert deciding_scores(scores, CFG) == scores


def test_summary_score_without_config_counts_everything() -> None:
    """Старый вызов без конфига обязан вести себя как раньше."""
    assert quality_score({"blur": 0.4, "noise": 0.1}) == 0.6
