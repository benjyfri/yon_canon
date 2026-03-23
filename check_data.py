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
    # Replaced 'distance_from_coordinates' with the correct 'distance_from_point'
    distances = [hilbert_curve.distance_from_point(pt) for pt in pc_int]

    return np.argsort(distances)


def compute_displacements(clean_sorted_idx: np.ndarray, perturbed_sorted_idx: np.ndarray):
    """
    Calculates how far points shifted in the sorted 1D array.
    Returns (average_displacement, median_displacement).
    """
    # argsort on the sorted indices gives the rank (position) of each original point
    clean_ranks = np.argsort(clean_sorted_idx)
    perturbed_ranks = np.argsort(perturbed_sorted_idx)

    displacements = np.abs(clean_ranks - perturbed_ranks)
    return np.mean(displacements), np.median(displacements)


# --- 3. Visualization ---

def plot_comprehensive_comparison(base_pc: np.ndarray, label: int):
    """Creates the 3x4 grid computing displacements dynamically."""
    high_res_size = base_pc.shape[0]
    low_res_size = 64

    # 1. Establish the clean subset and clean orderings
    fps_idx = get_fps_indices(base_pc, low_res_size)

    base_high = base_pc
    base_low = base_pc[fps_idx]

    lex_high_clean_idx = get_lexicographical_order_indices(base_high)
    hilb_high_clean_idx = get_hilbert_order_indices(base_high)

    lex_low_clean_idx = get_lexicographical_order_indices(base_low)
    hilb_low_clean_idx = get_hilbert_order_indices(base_low)

    scenarios = [
        ("Base", lambda pc: pc),
        ("Rotated 2°", lambda pc: apply_rotation(pc, 2.0)),
        ("Noisy", lambda pc: apply_noise(pc, 0.02))
    ]

    titles = []
    traces = []

    # 2. Pre-compute all transformations, indices, and displacement metrics
    for row_i, (scenario_name, transform_fn) in enumerate(scenarios, 1):
        trans_high = transform_fn(base_high.copy())
        # Apply the exact same FPS indices to maintain point identity
        trans_low = trans_high[fps_idx]

        # High-Res Lex
        idx_lex_h = get_lexicographical_order_indices(trans_high)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Lex ({high_res_size})")
        else:
            avg_d, med_d = compute_displacements(lex_high_clean_idx, idx_lex_h)
            # Replaced $\\Delta$ with the Unicode character Δ
            titles.append(f"{scenario_name} Lex ({high_res_size})<br>Avg Δ: {avg_d:.1f} | Med Δ: {med_d:.1f}")
        traces.append((trans_high[idx_lex_h], 'Viridis', row_i, 1, 2))

        # High-Res Hilbert
        idx_hilb_h = get_hilbert_order_indices(trans_high)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Hilbert ({high_res_size})")
        else:
            avg_d, med_d = compute_displacements(hilb_high_clean_idx, idx_hilb_h)
            # Replaced $\\Delta$ with the Unicode character Δ
            titles.append(f"{scenario_name} Hilbert ({high_res_size})<br>Avg Δ: {avg_d:.1f} | Med Δ: {med_d:.1f}")
        traces.append((trans_high[idx_hilb_h], 'Plasma', row_i, 2, 2))

        # Low-Res Lex (64 points)
        idx_lex_l = get_lexicographical_order_indices(trans_low)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Lex ({low_res_size})")
        else:
            avg_d, med_d = compute_displacements(lex_low_clean_idx, idx_lex_l)
            # Replaced $\\Delta$ with the Unicode character Δ
            titles.append(f"{scenario_name} Lex ({low_res_size})<br>Avg Δ: {avg_d:.1f} | Med Δ: {med_d:.1f}")
        traces.append((trans_low[idx_lex_l], 'Viridis', row_i, 3, 5))

        # Low-Res Hilbert (64 points)
        idx_hilb_l = get_hilbert_order_indices(trans_low)
        if scenario_name == "Base":
            titles.append(f"{scenario_name} Hilbert ({low_res_size})")
        else:
            avg_d, med_d = compute_displacements(hilb_low_clean_idx, idx_hilb_l)
            # Replaced $\\Delta$ with the Unicode character Δ
            titles.append(f"{scenario_name} Hilbert ({low_res_size})<br>Avg Δ: {avg_d:.1f} | Med Δ: {med_d:.1f}")
        traces.append((trans_low[idx_hilb_l], 'Plasma', row_i, 4, 5))

    # 3. Setup the Plotly Figure
    fig = make_subplots(
        rows=3, cols=4,
        specs=[[{'type': 'scatter3d'}] * 4] * 3,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.01
    )

    for pc, colorscale, row, col, marker_size in traces:
        add_trace(fig, pc, colorscale, row, col, marker_size)

    # Clean up layout
    axis_settings = dict(showbackground=False, showticklabels=False, title='')
    scene_settings = dict(xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings, aspectmode='data')
    scenes = {f'scene{i}': scene_settings for i in range(1, 13)}

    fig.update_layout(
        title=dict(
            text=f"Canonization Stress Test (Class: {label})<br>Measuring Array Displacement (Δ) vs Clean Base",
            x=0.5),
        margin=dict(l=0, r=0, b=0, t=80),
        showlegend=False,
        height=1300,
        width=1800,
        **scenes
    )

    output_filename = f"lex_vs_hil_{label}.html"
    fig.write_html(output_filename)
    print(f"Saved interactive plot to {os.path.abspath(output_filename)}")
    fig.show()


def add_trace(fig, pc, colorscale, row, col, marker_size):
    """Helper function to add a 3D scatter trace to a specific subplot."""
    indices = np.arange(pc.shape[0])
    fig.add_trace(
        go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size,  # Dynamic size: 2 for high-res, 5 for low-res
                color=indices,
                colorscale=colorscale,
                opacity=0.9
            ),
        ),
        row=row, col=col
    )


# --- 4. Execution ---

if __name__ == "__main__":
    base_dir = "data"
    raw_h5_file = os.path.join(base_dir, "modelnet40_ply_hdf5_2048", "ply_data_train0.h5")

    if not os.path.exists(raw_h5_file):
        print(f"Error: Could not find {raw_h5_file}. Ensure the path is correct.")
    else:
        with h5py.File(raw_h5_file, 'r') as f:
            for i in [90,190, 290, 390]:
                raw_pc = f['data'][i]
                label = f['label'][i][0]
                print(f"Processing shape index {i} (Label: {label}). Computing metrics...")
                plot_comprehensive_comparison(raw_pc, label)