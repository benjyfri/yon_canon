import os
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_orderings(lex_h5_path: str, hilbert_h5_path: str, shape_idx: int = 0):
    # 1. Load the data
    print(f"Loading shape index {shape_idx}...")
    with h5py.File(lex_h5_path, 'r') as f_lex:
        lex_pc = f_lex['data'][shape_idx]  # Shape: (2048, 3)
        label = f_lex['label'][shape_idx][0]

    with h5py.File(hilbert_h5_path, 'r') as f_hilb:
        hilb_pc = f_hilb['data'][shape_idx]  # Shape: (2048, 3)

    num_points = lex_pc.shape[0]
    indices = np.arange(num_points)  # This represents the order (0 to 2047)

    # 2. Setup Plotly Side-by-Side Figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            f"Lexicographical Order (Viridis)",
            f"Hilbert Curve Order (Plasma)"
        )
    )

    # 3. Add Lexicographical Trace (Left)
    fig.add_trace(
        go.Scatter3d(
            x=lex_pc[:, 0], y=lex_pc[:, 1], z=lex_pc[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=indices,  # Map color to the array index
                colorscale='Viridis',  # Requested colorscale
                opacity=0.9,
                colorbar=dict(title="Lex Order", x=0.45, thickness=15)
            ),
            name='Lex'
        ),
        row=1, col=1
    )

    # 4. Add Hilbert Trace (Right)
    fig.add_trace(
        go.Scatter3d(
            x=hilb_pc[:, 0], y=hilb_pc[:, 1], z=hilb_pc[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=indices,  # Map color to the array index
                colorscale='Plasma',  # Requested colorscale
                opacity=0.9,
                colorbar=dict(title="Hilbert Order", x=1.0, thickness=15)
            ),
            name='Hilbert'
        ),
        row=1, col=2
    )

    # 5. Lock aspect ratio so the 3D shapes aren't squished
    axis_settings = dict(showbackground=False, showticklabels=False, title='')
    scene_settings = dict(
        xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings,
        aspectmode='data'
    )

    fig.update_layout(
        title=dict(text=f"Point Cloud Ordering Comparison (Class Label: {label})", x=0.5),
        scene=scene_settings,
        scene2=scene_settings,
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False
    )

    # 6. Save the interactive HTML plot
    output_filename = "airplane_ordering_comparison.html"
    fig.write_html(output_filename)
    print(f"Saved interactive plot to {os.path.abspath(output_filename)}")

    # Optional: You can still have it open automatically right after saving
    print("Opening plot in your web browser...")
    fig.show()


if __name__ == "__main__":
    # Point these to the first train files in your new directories
    # (Adjust the path if you run this from a different folder level)
    base_dir = "data"
    lex_file = os.path.join(base_dir, "modelnet40_lex_hdf5_2048", "ply_data_train0.h5")
    hilbert_file = os.path.join(base_dir, "modelnet40_hilbert_hdf5_2048", "ply_data_train0.h5")

    if not os.path.exists(lex_file) or not os.path.exists(hilbert_file):
        print("Error: Could not find the .h5 files. Make sure the paths are correct!")
    else:
        # You can change shape_idx to visualize different objects in the h5 file
        for i in [140]:
            visualize_orderings(lex_file, hilbert_file, shape_idx=i)