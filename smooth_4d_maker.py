import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom

patient_data = np.load("/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/patient002.npy_segmentation_data.npy")

z_factor = 5 
interpolated_data = zoom(patient_data, (1, z_factor, 1, 1), order=1) 
print(f"Interpolated shape: {interpolated_data.shape}")

frames_dict = {frame_idx: interpolated_data[frame_idx] for frame_idx in range(interpolated_data.shape[0])}

num_frames = interpolated_data.shape[0]
num_slices = interpolated_data.shape[1]

fig = go.Figure()

fig.add_traces([
    go.Surface(
        z=slice_idx * np.ones_like(interpolated_data[0][slice_idx]),
        surfacecolor=interpolated_data[0][slice_idx],
        #colorscale="Viridis",
        showscale=(slice_idx == 0),  
        opacity=0.1
    )
    for slice_idx in range(num_slices)
])

fig.frames = [
    go.Frame(
        data=[
            go.Surface(
                z=slice_idx * np.ones_like(interpolated_data[time_step][slice_idx]),
                surfacecolor=interpolated_data[time_step][slice_idx],
                colorscale="Viridis",
                showscale=False,  
                opacity=0.1
            )
            for slice_idx in range(num_slices)
        ],
        name=f"time_{time_step}"
    )
    for time_step in range(num_frames)
]

fig.update_layout(
    title=f'4D Visualization for Patient 1',
    width=700,
    height=700,
    margin=dict(l=65, r=50, b=65, t=90),
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Slice Index"
    ),
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "fromcurrent": True}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time Step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {"args": [[f"time_{time_step}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
             "label": str(time_step),
             "method": "animate"}
            for time_step in range(num_frames)
        ]
    }]
)

fig.show()
