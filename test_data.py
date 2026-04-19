import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import OrderedModelNet40  # Assumes your file is named data.py


CACHE_FILE_512 = "data/modelnet40_ply_hdf5_2048/fps_cache/train_fps_512.npz"


def lexsort_points(pc: np.ndarray) -> np.ndarray:
    """Lexicographically sort points by x, then y, then z."""
    return pc[np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0]))]


def test_stride_logic():
    print("\n[1/12] Testing Dataset Stride (Subset Selection)...")
    dataset_full = OrderedModelNet40(1024, dataset_stride=1, use_fps=False)
    dataset_subset = OrderedModelNet40(1024, dataset_stride=5, use_fps=False)

    expected_len = len(range(0, len(dataset_full), 5))
    assert len(dataset_subset) == expected_len, f"Expected {expected_len}, got {len(dataset_subset)}"
    print("✅ Stride successfully reduces dataset size.")


def test_fallback_downsampling():
    print("\n[2/12] Testing Non-FPS (Stride-based) Downsampling...")
    dataset = OrderedModelNet40(512, dataset_stride=100, use_fps=False)
    pc, label = dataset[0]

    assert pc.shape == (512, 3), f"Expected shape (512, 3), got {pc.shape}"
    assert label is not None
    print("✅ Fallback downsampling successfully crops to exactly num_points.")


def test_fps_caching_and_shapes():
    print("\n[3/12] Testing FPS Downsampling and Cache Generation...")
    if os.path.exists(CACHE_FILE_512):
        os.remove(CACHE_FILE_512)

    start_time = time.time()
    _ = OrderedModelNet40(512, dataset_stride=100, use_fps=True)
    gen_time = time.time() - start_time

    assert os.path.exists(CACHE_FILE_512), "Cache file was not created!"

    mtime_before = os.path.getmtime(CACHE_FILE_512)

    start_time = time.time()
    dataset_load = OrderedModelNet40(512, dataset_stride=100, use_fps=True)
    load_time = time.time() - start_time

    mtime_after = os.path.getmtime(CACHE_FILE_512)
    assert mtime_before == mtime_after, "Cache file was regenerated instead of reused."

    pc, _ = dataset_load[0]
    assert pc.shape == (512, 3), f"Expected shape (512, 3), got {pc.shape}"
    assert pc.dtype == np.float32, f"Expected float32, got {pc.dtype}"

    print(f"✅ FPS caching works. Generation took {gen_time:.2f}s, cache load took {load_time:.2f}s.")


def test_random_permutation():
    print("\n[4/12] Testing Random Permutation (Always ON)...")
    dataset = OrderedModelNet40(
        512,
        dataset_stride=100,
        use_fps=False,
        apply_jitter=False,
        apply_anisotropic_scale=False,
    )

    pc1, _ = dataset[0]
    pc2, _ = dataset[0]

    exact_match = np.array_equal(pc1, pc2)
    if exact_match:
        pc3, _ = dataset[0]
        exact_match = np.array_equal(pc1, pc3)

    assert not exact_match, "Points were not permuted! Repeated calls returned identical order."

    pc1_sorted = lexsort_points(pc1)
    pc2_sorted = lexsort_points(pc2)
    assert np.array_equal(pc1_sorted, pc2_sorted), "Permutation altered the actual coordinate values!"
    print("✅ Random permutation shuffles order but preserves exact coordinate values.")


def test_train_augmentations():
    print("\n[5/12] Testing Train-Time Augmentations (Jitter & Scale)...")
    ds_base = OrderedModelNet40(
        512,
        dataset_stride=100,
        use_fps=False,
        apply_jitter=False,
        apply_anisotropic_scale=False,
    )
    ds_aug = OrderedModelNet40(
        512,
        dataset_stride=100,
        use_fps=False,
        apply_jitter=True,
        apply_anisotropic_scale=True,
    )

    pc_base, _ = ds_base[0]
    pc_aug, _ = ds_aug[0]

    base_extent = np.ptp(pc_base, axis=0)
    aug_extent = np.ptp(pc_aug, axis=0)

    assert not np.allclose(base_extent, aug_extent), "Augmentations did not change the point cloud geometry!"
    assert pc_aug.dtype == np.float32, f"Expected float32, got {pc_aug.dtype}"
    assert np.isfinite(pc_aug).all(), "Augmented point cloud contains NaN or inf"

    print("✅ Jitter and scale successfully modify the point cloud geometry.")


def test_partition_safety():
    print("\n[6/12] Testing Partition Safety (No Augs in Test/Val)...")
    ds_test = OrderedModelNet40(
        512,
        partition='test',
        dataset_stride=100,
        use_fps=False,
        apply_jitter=True,
        apply_anisotropic_scale=True,
    )

    pc1, _ = ds_test[0]
    pc2, _ = ds_test[0]

    pc1_sorted = lexsort_points(pc1)
    pc2_sorted = lexsort_points(pc2)

    assert np.array_equal(pc1_sorted, pc2_sorted), "SAFETY BREACH: Augmentations were applied to the test set!"
    print("✅ Partition safety verified. Augmentations are ignored outside 'train'.")


def test_dataloader():
    print("\n[7/12] Testing PyTorch DataLoader Integration...")
    dataset = OrderedModelNet40(512, dataset_stride=100, use_fps=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch_pc, batch_labels = next(iter(loader))
    assert batch_pc.shape == (8, 512, 3), f"Wrong batch shape: {batch_pc.shape}"
    assert batch_pc.dtype == torch.float32, "Point cloud is not float32!"
    assert batch_labels.dtype == torch.int64, "Labels are not int64!"
    print("✅ PyTorch DataLoader successfully collates batches.")


def test_no_inplace_mutation():
    print("\n[8/12] Testing No In-Place Mutation of Cached Data...")
    ds = OrderedModelNet40(
        512,
        dataset_stride=100,
        use_fps=False,
        apply_jitter=True,
        apply_anisotropic_scale=True,
    )

    before = ds.data[0].copy()
    _pc, _ = ds[0]
    after = ds.data[0]

    assert np.array_equal(before, after), "Dataset storage was mutated by __getitem__!"
    print("✅ Cached dataset remains unchanged after augmentation.")


def test_dtype_and_finiteness():
    print("\n[9/12] Testing Output Dtype and Numerical Stability...")
    ds = OrderedModelNet40(
        512,
        dataset_stride=100,
        use_fps=False,
        apply_jitter=True,
        apply_anisotropic_scale=True,
    )

    pc, label = ds[0]

    assert pc.dtype == np.float32, f"Expected float32, got {pc.dtype}"
    assert np.isfinite(pc).all(), "Point cloud contains NaN or inf"
    assert np.issubdtype(type(label.item()), np.integer), f"Label is not integer-like: {type(label)}"
    print("✅ Output dtype and finiteness are correct.")


def test_exact_point_count_all_paths():
    print("\n[10/12] Testing Exact Point Count Across Retrieval Paths...")
    for use_fps in [False, True]:
        ds = OrderedModelNet40(512, dataset_stride=100, use_fps=use_fps)
        pc, _ = ds[0]
        assert pc.shape == (512, 3), f"Bad shape for use_fps={use_fps}: {pc.shape}"
    print("✅ Both FPS and non-FPS paths return exactly num_points.")


def test_cache_reuse_without_regeneration():
    print("\n[11/12] Testing Cache Reuse Without Regeneration...")
    if os.path.exists(CACHE_FILE_512):
        os.remove(CACHE_FILE_512)

    _ = OrderedModelNet40(512, dataset_stride=100, use_fps=True)
    assert os.path.exists(CACHE_FILE_512), "Cache file was not created."

    mtime1 = os.path.getmtime(CACHE_FILE_512)
    _ = OrderedModelNet40(512, dataset_stride=100, use_fps=True)
    mtime2 = os.path.getmtime(CACHE_FILE_512)

    assert mtime1 == mtime2, "Cache file was regenerated instead of reused."
    print("✅ Cache file is reused without regeneration.")


def test_dataloader_multiworker():
    print("\n[12/12] Testing Multi-Worker DataLoader Integration...")
    ds = OrderedModelNet40(512, dataset_stride=100, use_fps=False)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=False)

    batch_pc, batch_labels = next(iter(loader))
    assert batch_pc.shape == (4, 512, 3), f"Wrong batch shape: {batch_pc.shape}"
    assert batch_pc.dtype == torch.float32, "Point cloud is not float32!"
    assert batch_labels.dtype == torch.int64, "Labels are not int64!"
    print("✅ Multi-worker DataLoader works correctly.")


if __name__ == "__main__":
    print("🚀 Starting Rigorous Testing Suite for OrderedModelNet40...\n")
    try:
        test_stride_logic()
        test_fallback_downsampling()
        test_fps_caching_and_shapes()
        test_random_permutation()
        test_train_augmentations()
        test_partition_safety()
        test_dataloader()
        test_no_inplace_mutation()
        test_dtype_and_finiteness()
        test_exact_point_count_all_paths()
        test_cache_reuse_without_regeneration()
        test_dataloader_multiworker()
        print("\n🎉 ALL TESTS PASSED! Your data.py is in very good shape.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise