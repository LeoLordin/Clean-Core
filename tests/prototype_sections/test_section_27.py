import numpy as np
import pytest

from clean_core.prototypes.sections.section_27 import (
    ConformalTheoryOfMindPlanner,
    conformalize_belief,
)


def test_conformal_belief_is_normalized_and_shrinks_toward_uniform():
    transformed = conformalize_belief([1.0, 3.0], uniform_shrinkage=0.5)

    assert transformed.shape == (2,)
    assert np.all((0.0 <= transformed) & (transformed <= 1.0))
    np.testing.assert_allclose(transformed, [0.375, 0.625])
    assert transformed.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(conformalize_belief([1.0, 3.0], 1.0), [0.5, 0.5])


def test_action_choice_uses_multi_agent_expected_utilities_and_normalized_beliefs():
    planner = ConformalTheoryOfMindPlanner()
    utilities = np.array(
        [
            [[4.0, 0.0], [0.0, 2.0]],
            [[0.0, 6.0], [3.0, 0.0]],
        ]
    )

    action, scores = planner.choose_action(
        [0.0, 0.1],
        [[1.0, 3.0], [3.0, 1.0]],
        utilities,
    )

    np.testing.assert_allclose(scores, [1.5, 6.85])
    assert action == 1


def test_scores_clip_curiosity_and_utility_to_configured_bound():
    planner = ConformalTheoryOfMindPlanner(score_bound=2.0)
    scores = planner.score_actions([10.0], [[1.0]], np.array([[[10.0]]]))

    np.testing.assert_allclose(scores, [4.0])


@pytest.mark.parametrize(
    "belief",
    [
        [],
        [0.0, 0.0],
        [-1.0, 2.0],
        [1.0, np.nan],
        [[1.0, 2.0]],
    ],
)
def test_conformal_belief_rejects_invalid_distributions(belief):
    with pytest.raises(ValueError, match="action_distribution"):
        conformalize_belief(belief)


@pytest.mark.parametrize("shrinkage", [-0.1, 1.1, np.nan])
def test_conformal_belief_rejects_invalid_shrinkage(shrinkage):
    with pytest.raises(ValueError, match="uniform_shrinkage"):
        conformalize_belief([1.0, 1.0], shrinkage)


@pytest.mark.parametrize(
    "curiosity, beliefs, utilities",
    [
        ([0.0, 1.0], [[1.0]], np.zeros((1, 1, 1))),
        ([0.0], [[1.0]], np.zeros((1, 1))),
        ([0.0], [[1.0, 2.0]], np.zeros((1, 1, 1))),
        ([0.0], [[1.0]], np.zeros((1, 1, 0))),
    ],
)
def test_planner_rejects_shape_mismatches(curiosity, beliefs, utilities):
    with pytest.raises(ValueError):
        ConformalTheoryOfMindPlanner().score_actions(curiosity, beliefs, utilities)
