import os
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hilbertcurve.hilbertcurve import HilbertCurve


# --- 1. Core Geometric & Sampling Functions ---

def get_fps_indices(pc: np.ndarray, num_points: int):
    """Returns the indices for Farthest Point Sampling (FPS)."""
    N = pc.shape[0]
    if num_points >= N:
        return np.arange(N)

    fps_idx = np.zeros(num_points, dtype=int)
    distances = np.ones(N) * 1e10

    fps_idx[0] = np.random.randint(0, N)

    for i in range(1, num_points):
        dist_to_last = np.sum((pc - pc[fps_idx[i - 1]]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        fps_idx[i] = np.argmax(distances)

    return fps_idx


def apply_rotation(pc: np.ndarray, degrees: float = 2.0):
    """Applies a slight rotation around the Z and Y axes."""
    theta = np.radians(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    Rz = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

    R = Rz @ Ry
    return pc @ R.T


def apply_noise(pc: np.ndarray, std: float = 0.02):
    """Adds slight Gaussian noise to the point cloud."""
    noise = np.random.normal(0, std, pc.shape)
    return pc + noise


# --- 2. Dynamic Canonization & Tracking Functions ---

def get_lexicographical_order_indices(pc: np.ndarray):
    """Returns the sorting indices for Lexicographical order (X, Y, Z)."""
    return np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0]))


def get_hilbert_order_indices(pc: np.ndarray, p: int = 10):
    """Returns the sorting indices for Hilbert Curve order."""
    pc_min = np.min(pc, axis=0)
    pc_max = np.max(pc, axis=0)
    pc_norm = (pc - pc_min) / (pc_max - pc_min + 1e-8)

    max_coord = (1 << p) - 1
    pc_int = np.clip(np.floor(pc_norm * max_coord), 0, max_coord).astype(int)

    hilbert_curve = HilbertCurve(p, 3)
    distances = [hilbert_curve.distance_from_point(pt) for pt in pc_int]

    return np.argsort(distances)


def compute_displacements(clean_sorted_idx: np.ndarray, perturbed_sorted_idx: np.ndarray):
    """
    Calculates how far points shifted in the sorted 1D array.
    Returns (average_displacement, median_displacement, max_displacement).
    """
    clean_ranks = np.argsort(clean_sorted_idx)
    perturbed_ranks = np.argsort(perturbed_sorted_idx)

    displacements = np.abs(clean_ranks - perturbed_ranks)
    return np.mean(displacements), np.median(displacements), np.max(displacements)


# --- 3. Dataset Statistics Function ---

def compute_dataset_statistics(h5_file_path: str, low_res_size: int = 64):
    """
    Iterates over all point clouds in the dataset to compute average, median,
    and max array displacements across different orderings and perturbations.
    """
    stats = {
        "Rotated_Lex_High": {"avg": [], "med": [], "max": []},
        "Rotated_Hilb_High": {"avg": [], "med": [], "max": []},
        "Rotated_Lex_Low": {"avg": [], "med": [], "max": []},
        "Rotated_Hilb_Low": {"avg": [], "med": [], "max": []},
        "Noisy_Lex_High": {"avg": [], "med": [], "max": []},
        "Noisy_Hilb_High": {"avg": [], "med": [], "max": []},
        "Noisy_Lex_Low": {"avg": [], "med": [], "max": []},
        "Noisy_Hilb_Low": {"avg": [], "med": [], "max": []},
    }

    print(f"Loading dataset from {h5_file_path}...")
    with h5py.File(h5_file_path, 'r') as f:
        dataset = f['data']
        num_shapes = dataset.shape[0]
        high_res_size = dataset.shape[1]

        print(f"Found {num_shapes} shapes. Starting bulk computation...\n")

        for i in range(num_shapes):
            base_high = dataset[i]

            # 1. Establish the clean subset and clean orderings
            fps_idx = get_fps_indices(base_high, low_res_size)
            base_low = base_high[fps_idx]

            lex_high_clean_idx = get_lexicographical_order_indices(base_high)
            hilb_high_clean_idx = get_hilbert_order_indices(base_high)

            lex_low_clean_idx = get_lexicographical_order_indices(base_low)
            hilb_low_clean_idx = get_hilbert_order_indices(base_low)

            # 2. Define Transformations
            scenarios = [
                ("Rotated", lambda pc: apply_rotation(pc, 2.0)),
                ("Noisy", lambda pc: apply_noise(pc, 0.02))
            ]

            # 3. Compute and store metrics for each scenario
            for prefix, transform_fn in scenarios:
                trans_high = transform_fn(base_high.copy())
                trans_low = trans_high[fps_idx]

                # High-Res
                a, m, mx = compute_displacements(lex_high_clean_idx, get_lexicographical_order_indices(trans_high))
                stats[f"{prefix}_Lex_High"]["avg"].append(a)
                stats[f"{prefix}_Lex_High"]["med"].append(m)
                stats[f"{prefix}_Lex_High"]["max"].append(mx)

                a, m, mx = compute_displacements(hilb_high_clean_idx, get_hilbert_order_indices(trans_high))
                stats[f"{prefix}_Hilb_High"]["avg"].append(a)
                stats[f"{prefix}_Hilb_High"]["med"].append(m)
                stats[f"{prefix}_Hilb_High"]["max"].append(mx)

                # Low-Res
                a, m, mx = compute_displacements(lex_low_clean_idx, get_lexicographical_order_indices(trans_low))
                stats[f"{prefix}_Lex_Low"]["avg"].append(a)
                stats[f"{prefix}_Lex_Low"]["med"].append(m)
                stats[f"{prefix}_Lex_Low"]["max"].append(mx)

                a, m, mx = compute_displacements(hilb_low_clean_idx, get_hilbert_order_indices(trans_low))
                stats[f"{prefix}_Hilb_Low"]["avg"].append(a)
                stats[f"{prefix}_Hilb_Low"]["med"].append(m)
                stats[f"{prefix}_Hilb_Low"]["max"].append(mx)

            # Progress tracker
            if (i + 1) % 200 == 0 or (i + 1) == num_shapes:
                print(f"Processed {i + 1}/{num_shapes} shapes...")

    # 4. Print Summary Table
    print("\n" + "=" * 80)
    print(f"DATASET DISPLACEMENT STATISTICS (Averaged over {num_shapes} shapes)")
    print("=" * 80)
    print(f"{'Scenario / Ordering':<30} | {'Mean Δ':<10} | {'Median Δ':<10} | {'Max Δ':<10}")
    print("-" * 80)

    for key, values in stats.items():
        grand_avg = np.mean(values["avg"])
        grand_med = np.mean(values["med"])
        grand_max = np.mean(values["max"])

        print(f"{key:<30} | {grand_avg:<10.1f} | {grand_med:<10.1f} | {grand_max:<10.1f}")

    print("=" * 80)


# --- 4. Visualization Function (Retained for completeness) ---

def plot_comprehensive_comparison(base_pc: np.ndarray, label: int):
    """Creates the 3x4 grid computing displacements dynamically."""
    high_res_size = base_pc.shape[0]
    low_res_size = 64

    fps_idx = get_fps_indices(base_pc, low_res_size)
    base_low = base_pc[fps_idx]

    lex_high_clean_idx = get_lexicographical_order_indices(base_pc)
    hilb_high_clean_idx = get_hilbert_order_indices(base_pc)
    lex_low_clean_idx = get_lexicographical_order_indices(base_low)
    hilb_low_clean_idx = get_hilbert_order_indices(base_low)

    scenarios = [
        ("Base", lambda pc: pc),
        ("Rotated 2°", lambda pc: apply_rotation(pc, 2.0)),
        ("Noisy", lambda pc: apply_noise(pc, 0.02))
    ]

    titles = []
    traces = []

    for row_i, (scenario_name, transform_fn) in enumerate(scenarios, 1):
        trans_high = transform_fn(base_pc.copy())
        trans_low = trans_high[fps_idx]

        # High-Res Lex
        idx_lex_h = get_lexicographical_order_indices(trans_high)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Lex ({high_res_size})")
        else:
            a, m, mx = compute_displacements(lex_high_clean_idx, idx_lex_h)
            titles.append(f"{scenario_name} Lex ({high_res_size})<br>Avg Δ: {a:.1f} | Med Δ: {m:.1f} | Max Δ: {mx:.1f}")
        traces.append((trans_high[idx_lex_h], 'Viridis', row_i, 1, 2))

        # High-Res Hilbert
        idx_hilb_h = get_hilbert_order_indices(trans_high)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Hilbert ({high_res_size})")
        else:
            a, m, mx = compute_displacements(hilb_high_clean_idx, idx_hilb_h)
            titles.append(
                f"{scenario_name} Hilbert ({high_res_size})<br>Avg Δ: {a:.1f} | Med Δ: {m:.1f} | Max Δ: {mx:.1f}")
        traces.append((trans_high[idx_hilb_h], 'Plasma', row_i, 2, 2))

        # Low-Res Lex
        idx_lex_l = get_lexicographical_order_indices(trans_low)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Lex ({low_res_size})")
        else:
            a, m, mx = compute_displacements(lex_low_clean_idx, idx_lex_l)
            titles.append(f"{scenario_name} Lex ({low_res_size})<br>Avg Δ: {a:.1f} | Med Δ: {m:.1f} | Max Δ: {mx:.1f}")
        traces.append((trans_low[idx_lex_l], 'Viridis', row_i, 3, 5))

        # Low-Res Hilbert
        idx_hilb_l = get_hilbert_order_indices(trans_low)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Hilbert ({low_res_size})")
        else:
            a, m, mx = compute_displacements(hilb_low_clean_idx, idx_hilb_l)
            titles.append(
                f"{scenario_name} Hilbert ({low_res_size})<br>Avg Δ: {a:.1f} | Med Δ: {m:.1f} | Max Δ: {mx:.1f}")
        traces.append((trans_low[idx_hilb_l], 'Plasma', row_i, 4, 5))

    fig = make_subplots(
        rows=3, cols=4, specs=[[{'type': 'scatter3d'}] * 4] * 3,
        subplot_titles=titles, vertical_spacing=0.08, horizontal_spacing=0.01
    )

    for pc, colorscale, row, col, marker_size in traces:
        add_trace(fig, pc, colorscale, row, col, marker_size)

    axis_settings = dict(showbackground=False, showticklabels=False, title='')
    scene_settings = dict(xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings, aspectmode='data')
    scenes = {f'scene{i}': scene_settings for i in range(1, 13)}

    fig.update_layout(
        title=dict(text=f"Canonization Stress Test (Class: {label})", x=0.5),
        margin=dict(l=0, r=0, b=0, t=80), showlegend=False,
        height=1300, width=1800, **scenes
    )

    output_filename = "comprehensive_displacement_stress_test.html"
    fig.write_html(output_filename)
    print(f"Saved interactive plot to {os.path.abspath(output_filename)}")
    fig.show()


def add_trace(fig, pc, colorscale, row, col, marker_size):
    indices = np.arange(pc.shape[0])
    fig.add_trace(
        go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(size=marker_size, color=indices, colorscale=colorscale, opacity=0.9),
        ),
        row=row, col=col
    )


# --- 5. Execution ---

if __name__ == "__main__":
    base_dir = "data"
    raw_h5_file = os.path.join(base_dir, "modelnet40_ply_hdf5_2048", "ply_data_train0.h5")

    if not os.path.exists(raw_h5_file):
        print(f"Error: Could not find {raw_h5_file}. Ensure the path is correct.")
    else:
        # Toggles set to only run the bulk statistics over the dataset
        RUN_VISUALIZATION = False
        RUN_BULK_STATS = True

        if RUN_VISUALIZATION:
            with h5py.File(raw_h5_file, 'r') as f:
                shape_idx = 90
                raw_pc = f['data'][shape_idx]
                label = f['label'][shape_idx][0]
                print(f"Processing shape index {shape_idx} (Label: {label}) for visualization...")
                plot_comprehensive_comparison(raw_pc, label)

        if RUN_BULK_STATS:
            compute_dataset_statistics(raw_h5_file, low_res_size=64)