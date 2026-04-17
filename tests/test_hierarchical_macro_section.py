import numpy as np

from src.hierarchical import select_initial_match_components


def _seed_inputs():
    geodesic = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    return geodesic, edge, centroids


def test_select_initial_match_components_prefers_connected_three_node_chain():
    matches = [(0, 0), (1, 1), (2, 2), (0, 1)]
    match_adj = np.array(
        [
            [False, True, False, False],
            [True, False, True, False],
            [False, True, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    match_scores = np.array([2.0, 2.0, 2.0, 0.5], dtype=np.float64)
    match_tiebreak_scores = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    geodesic, edge, centroids = _seed_inputs()

    selected, diagnostics = select_initial_match_components(
        matches=matches,
        match_adj=match_adj,
        match_scores=match_scores,
        geodesic_A=geodesic,
        geodesic_B=geodesic,
        edge_A_norm=edge,
        edge_B_norm=edge,
        edge_scale_A=1.0,
        edge_scale_B=1.0,
        centroids_A=centroids,
        centroids_B=centroids,
        match_tiebreak_scores=match_tiebreak_scores,
        top_k=2,
    )

    assert len(selected) >= 1
    np.testing.assert_array_equal(selected[0], np.array([0, 1, 2], dtype=int))
    assert diagnostics["seed_trial_sizes"][0] == 3


def test_select_initial_match_components_falls_back_to_singletons_when_multinode_scores_are_nonpositive():
    matches = [(0, 0), (1, 1)]
    match_adj = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )
    match_scores = np.array([-0.2, -0.2], dtype=np.float64)
    match_tiebreak_scores = np.array([0.1, 0.05], dtype=np.float64)
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    geodesic_A = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float64)
    geodesic_B = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    edge_A = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float64)
    edge_B = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    selected, diagnostics = select_initial_match_components(
        matches=matches,
        match_adj=match_adj,
        match_scores=match_scores,
        geodesic_A=geodesic_A,
        geodesic_B=geodesic_B,
        edge_A_norm=edge_A,
        edge_B_norm=edge_B,
        edge_scale_A=1.0,
        edge_scale_B=1.0,
        centroids_A=centroids,
        centroids_B=centroids,
        match_tiebreak_scores=match_tiebreak_scores,
        top_k=1,
    )

    assert len(selected) == 1
    assert selected[0].size == 1
    assert diagnostics["seed_search_mode"] == "singleton_fallback"


def test_select_initial_match_components_enforces_one_to_one_within_motif():
    matches = [(0, 0), (0, 1), (1, 2)]
    match_adj = np.array(
        [
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ],
        dtype=bool,
    )
    match_scores = np.array([3.0, 3.0, 2.0], dtype=np.float64)
    match_tiebreak_scores = np.array([0.3, 0.25, 0.2], dtype=np.float64)
    geodesic = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    centroids_A = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    centroids_B = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    selected, diagnostics = select_initial_match_components(
        matches=matches,
        match_adj=match_adj,
        match_scores=match_scores,
        geodesic_A=geodesic,
        geodesic_B=np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        edge_A_norm=edge,
        edge_B_norm=np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        edge_scale_A=1.0,
        edge_scale_B=1.0,
        centroids_A=centroids_A,
        centroids_B=centroids_B,
        match_tiebreak_scores=match_tiebreak_scores,
        top_k=1,
    )

    assert len(selected) == 1
    assert diagnostics["seed_trial_sizes"][0] == 2
    chosen_pairs = [matches[idx] for idx in selected[0]]
    assert len({u for u, _ in chosen_pairs}) == len(chosen_pairs)
    assert len({v for _, v in chosen_pairs}) == len(chosen_pairs)


def test_select_initial_match_components_penalizes_overlap_between_chosen_seeds():
    matches = [(i, i) for i in range(6)]
    match_adj = np.array(
        [
            [False, True, False, False, False, False],
            [True, False, True, False, False, False],
            [False, True, False, True, False, False],
            [False, False, True, False, False, False],
            [False, False, False, False, False, True],
            [False, False, False, False, True, False],
        ],
        dtype=bool,
    )
    match_scores = np.array([3.0, 3.0, 2.5, 2.0, 3.4, 3.4], dtype=np.float64)
    match_tiebreak_scores = np.array([0.4, 0.35, 0.3, 0.2, 0.5, 0.45], dtype=np.float64)
    geodesic = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 8.0, 9.0],
            [1.0, 0.0, 1.0, 2.0, 7.0, 8.0],
            [2.0, 1.0, 0.0, 1.0, 6.0, 7.0],
            [3.0, 2.0, 1.0, 0.0, 5.0, 6.0],
            [8.0, 7.0, 6.0, 5.0, 0.0, 1.0],
            [9.0, 8.0, 7.0, 6.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    centroids = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ],
        dtype=np.float64,
    )

    selected, diagnostics = select_initial_match_components(
        matches=matches,
        match_adj=match_adj,
        match_scores=match_scores,
        geodesic_A=geodesic,
        geodesic_B=geodesic,
        edge_A_norm=edge,
        edge_B_norm=edge,
        edge_scale_A=1.0,
        edge_scale_B=1.0,
        centroids_A=centroids,
        centroids_B=centroids,
        match_tiebreak_scores=match_tiebreak_scores,
        top_k=2,
    )

    assert len(selected) == 2
    np.testing.assert_array_equal(selected[0], np.array([0, 1, 2], dtype=int))
    np.testing.assert_array_equal(selected[1], np.array([4, 5], dtype=int))
    assert diagnostics["seed_diversification_mode"] == "greedy_overlap_penalized_topk"
    assert diagnostics["seed_trial_overlap_penalties"][1] == 0.0
