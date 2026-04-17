import unittest
from unittest.mock import patch
import sys
import types

import numpy as np
from anndata import AnnData

if "ot" not in sys.modules:
    ot_module = types.ModuleType("ot")
    backend_module = types.ModuleType("ot.backend")
    optim_module = types.ModuleType("ot.optim")
    gromov_module = types.ModuleType("ot.gromov")

    class DummyTorchBackend:
        pass

    class DummyNumpyBackend:
        pass

    backend_module.TorchBackend = DummyTorchBackend
    backend_module.NumpyBackend = DummyNumpyBackend
    backend_module.get_backend = lambda *args, **kwargs: DummyNumpyBackend()
    optim_module.line_search_armijo = lambda *args, **kwargs: None
    optim_module.cg = lambda *args, **kwargs: None
    gromov_module.solve_gromov_linesearch = lambda *args, **kwargs: None
    gromov_module.fused_unbalanced_gromov_wasserstein = (
        lambda *args, **kwargs: (np.zeros((1, 1), dtype=float), None, {})
    )

    ot_module.backend = backend_module
    ot_module.optim = optim_module
    ot_module.gromov = gromov_module
    ot_module.utils = types.SimpleNamespace(list_to_array=lambda *args: args)

    sys.modules["ot"] = ot_module
    sys.modules["ot.backend"] = backend_module
    sys.modules["ot.optim"] = optim_module
    sys.modules["ot.gromov"] = gromov_module

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch_module

from src import core, hierarchical


def make_adata(coords, label_key="cell_type_annot"):
    adata = AnnData(np.zeros((len(coords), 1), dtype=float))
    adata.obsm["spatial"] = np.asarray(coords, dtype=float)
    adata.obs[label_key] = np.array(["t0"] * len(coords), dtype=object)
    return adata


class MacroSectionTests(unittest.TestCase):
    def test_extract_continuous_macro_section_returns_empty_on_zero_mass(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        adata_a = make_adata(coords)
        adata_b = make_adata(coords)

        idx_a, idx_b, dist_a, dist_b, matched_pairs = hierarchical.extract_continuous_macro_section(
            adata_a,
            adata_b,
            labels_A=np.array([0, 0]),
            labels_B=np.array([0, 0]),
            Pi_cluster=np.zeros((1, 1), dtype=float),
        )

        self.assertEqual(idx_a.size, 0)
        self.assertEqual(idx_b.size, 0)
        self.assertEqual(matched_pairs.shape, (0, 2))
        self.assertTrue(np.all(dist_a == 0))
        self.assertTrue(np.all(dist_b == 0))

    def test_extract_continuous_macro_section_handles_non_dense_labels_and_collinear_geometry(self):
        coords_a = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [2.0, 0.0],
            [2.2, 0.0],
        ])
        coords_b = np.array([
            [0.0, 0.0],
            [0.0, 0.2],
            [0.0, 1.0],
            [0.0, 1.2],
            [0.0, 2.0],
            [0.0, 2.2],
        ])
        labels_a = np.array([10, 10, 20, 20, 30, 30])
        labels_b = np.array([100, 100, 200, 200, 300, 300])
        pi_cluster = np.array([
            [0.45, 0.02, 0.00],
            [0.01, 0.40, 0.02],
            [0.00, 0.02, 0.35],
        ])

        adata_a = make_adata(coords_a)
        adata_b = make_adata(coords_b)

        idx_a, idx_b, dist_a, dist_b, matched_pairs = hierarchical.extract_continuous_macro_section(
            adata_a,
            adata_b,
            labels_A=labels_a,
            labels_B=labels_b,
            Pi_cluster=pi_cluster,
        )

        np.testing.assert_array_equal(matched_pairs, np.array([[0, 0], [1, 1], [2, 2]]))
        self.assertEqual(idx_a.size, coords_a.shape[0])
        self.assertEqual(idx_b.size, coords_b.shape[0])
        self.assertTrue(np.all(dist_a[idx_a] == 0))
        self.assertTrue(np.all(dist_b[idx_b] == 0))

    def test_compute_cluster_structural_matrix_handles_collinear_centroids(self):
        centroids = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])

        matrix = hierarchical.compute_cluster_structural_matrix(centroids, w_euc=0.5, w_graph=0.5)

        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(matrix)))
        np.testing.assert_allclose(matrix, matrix.T)

    def test_hierarchical_pairwise_align_falls_back_to_direct_pairwise_alignment(self):
        label_key = "ct"
        adata_a = make_adata([[0.0, 0.0], [1.0, 0.0]], label_key=label_key)
        adata_b = make_adata([[0.0, 0.0], [1.0, 0.0]], label_key=label_key)
        sentinel = np.array([[1.0, 0.0], [0.0, 1.0]])
        feature_tuple = (
            np.array([1.0]),
            np.array([[0.0, 0.0]]),
            np.array([[0.0]]),
            np.array([[1.0]]),
        )

        with patch.object(core, "cluster_cells_spatial", side_effect=[np.array([0, 0]), np.array([0, 0])]):
            with patch.object(core, "extract_cluster_features", side_effect=[feature_tuple, feature_tuple]):
                with patch.object(core, "compute_cluster_feature_costs", return_value=np.zeros((1, 1))):
                    with patch.object(core, "compute_cluster_structural_matrix", return_value=np.zeros((1, 1))):
                        with patch.object(core, "run_coarse_partial_fgw", return_value=np.zeros((1, 1))):
                            with patch.object(core, "pairwise_align", return_value=sentinel) as pairwise_mock:
                                result = core.hierarchical_pairwise_align(
                                    adata_a,
                                    adata_b,
                                    alpha=0.5,
                                    beta=0.5,
                                    gamma=0.5,
                                    label_key=label_key,
                                    visualize_clusters=False,
                                )

        self.assertIs(result, sentinel)
        pairwise_mock.assert_called_once()
        self.assertEqual(pairwise_mock.call_args.kwargs["label_key"], label_key)
        self.assertEqual(pairwise_mock.call_args.kwargs["spatial_key"], "spatial")


if __name__ == "__main__":
    unittest.main()
