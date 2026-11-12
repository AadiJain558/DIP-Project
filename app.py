"""Streamlit demo: single 2-D image to approximate 3-D model."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageFilter
from skimage.filters import sobel
from skimage.restoration import denoise_bilateral
from transformers import AutoImageProcessor, AutoModelForDepthEstimation



st.set_page_config(page_title="Image 3D Reconstruction", layout="wide")

MODEL_ID = "Intel/dpt-hybrid-midas"

FILTER_DESCRIPTIONS: Dict[str, str] = {
	"Mean": "Softens noise by averaging pixels in a local neighborhood.",
	"Median": "Replaces pixels with the median of neighbors to remove salt & pepper noise.",
	"Gaussian": "Applies a Gaussian blur for gentle smoothing.",
	"Bilateral": "Smooths while preserving edges using spatial + color distance.",
	"Sharpen": "Boosts edge contrast via unsharp masking.",
	"Sobel": "Highlights intensity changes to expose edges.",
}


def _hash_image(image: Image.Image) -> bytes:
	"""Generate a content-based hash for Streamlit caching."""
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	return buf.getvalue()


def _pil_to_array(image: Image.Image) -> np.ndarray:
	return np.asarray(image).astype(np.float32) / 255.0


def _array_to_pil(arr: np.ndarray) -> Image.Image:
	arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
	return Image.fromarray(arr)


@st.cache_resource(show_spinner=False)
def load_depth_model():
	"""Load the Hugging Face DPT depth model once per session."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	try:
		processor = AutoImageProcessor.from_pretrained(MODEL_ID)
		model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
	except OSError as exc:
		raise RuntimeError(
			"Unable to download the depth model from Hugging Face.\n"
			"Check your internet connection or set HF_HOME to a cached model directory."
		) from exc
	model.to(device)
	model.eval()
	return processor, model, device


@st.cache_data(show_spinner=False, hash_funcs={Image.Image: _hash_image})
def estimate_depth_map(image: Image.Image) -> np.ndarray:
	"""Run the transformer depth model and return a normalized depth map."""
	processor, model, device = load_depth_model()
	inputs = processor(images=image, return_tensors="pt")
	inputs = {k: v.to(device) for k, v in inputs.items()}
	with torch.no_grad():
		prediction = model(**inputs).predicted_depth
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size=image.size[::-1],
			mode="bicubic",
			align_corners=False,
		).squeeze()
	depth = prediction.cpu().numpy()
	depth -= depth.min()
	depth /= depth.max() + 1e-8
	return depth


def sample_point_cloud(
	depth: np.ndarray,
	image: Image.Image,
	*,
	depth_scale: float = 1.5,
	max_points: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Convert depth map to a subsampled point cloud with RGB colors.

	The function creates a regular grid in normalized image coordinates,
	scales depth for visualization, and optionally subsamples points to
	limit the total count for interactive rendering.
	"""
	h, w = depth.shape
	xs = np.linspace(-1.0, 1.0, w)
	ys = np.linspace(-1.0, 1.0, h)
	grid_x, grid_y = np.meshgrid(xs, -ys)  # flip y for a natural camera view
	z = depth * depth_scale  # adjustable scaling for visual separation

	rgb = np.asarray(image.resize((w, h))).reshape(-1, 3) / 255.0
	points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), z.reshape(-1)))

	total = points.shape[0]
	if total > max_points:
		idx = np.linspace(0, total - 1, max_points, dtype=int)
		points = points[idx]
		rgb = rgb[idx]

	return points[:, 0], points[:, 1], points[:, 2], rgb


def render_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray) -> go.Figure:
	"""Build an interactive Plotly Scatter3D visualization."""
	fig = go.Figure(
		data=[
			go.Scatter3d(
				x=x,
				y=y,
				z=z,
				mode="markers",
				marker=dict(size=2, color=colors, opacity=0.8),
			)
		]
	)
	fig.update_layout(
		scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Depth"),
		margin=dict(l=0, r=0, t=0, b=0),
		height=600,
	)
	return fig


def point_cloud_to_ply(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray) -> bytes:
	"""Serialize the point cloud into a minimal ASCII PLY file."""
	header = """ply
format ascii 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".strip().format(count=len(x))

	body_lines = [
		f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {int(colors[i,0]*255)} {int(colors[i,1]*255)} {int(colors[i,2]*255)}"
		for i in range(len(x))
	]
	ply_bytes = "\n".join([header, *body_lines]).encode("utf-8")
	return ply_bytes


def apply_dip_filters(
	original: Image.Image,
	selected_filters: List[str],
	controls: Dict[str, float],
) -> List[Dict[str, object]]:
	"""Produce filtered previews and metadata for the selected filters."""
	results: List[Dict[str, object]] = []
	for name in selected_filters:
		if name == "Mean":
			radius = controls["mean_radius"]
			preview = original.filter(ImageFilter.BoxBlur(radius))
			params = f"radius={radius}"
		elif name == "Median":
			size = controls["median_kernel"]
			size = size if size % 2 else size + 1
			preview = original.filter(ImageFilter.MedianFilter(size=size))
			params = f"kernel={size}"
		elif name == "Gaussian":
			sigma = controls["gaussian_sigma"]
			preview = original.filter(ImageFilter.GaussianBlur(radius=sigma))
			params = f"sigma={sigma:.1f}"
		elif name == "Bilateral":
			sigma_color = controls["bilateral_color"]
			sigma_space = controls["bilateral_space"]
			arr = _pil_to_array(original)
			filtered = denoise_bilateral(
				arr,
				sigma_color=sigma_color,
				sigma_spatial=sigma_space,
				channel_axis=-1,
			)
			preview = _array_to_pil(filtered)
			params = f"σ_color={sigma_color:.2f}, σ_space={sigma_space:.2f}"
		elif name == "Sharpen":
			amount = controls["sharpen_amount"]
			preview = original.filter(
				ImageFilter.UnsharpMask(radius=2, percent=int(amount * 100), threshold=5)
			)
			params = f"amount={amount:.2f}"
		elif name == "Sobel":
			gray = original.convert("L")
			arr = np.asarray(gray).astype(np.float32) / 255.0
			edges = sobel(arr)
			if edges.max() > 0:
				edges /= edges.max()
			edges_rgb = np.stack([edges] * 3, axis=-1)
			preview = _array_to_pil(edges_rgb)
			params = "normalized"
		else:
			continue
		results.append(
			{
				"Filter": name,
				"Description": FILTER_DESCRIPTIONS[name],
				"Parameters": params,
				"Preview": preview,
			}
		)
	return results


def main() -> None:
	st.title("Single Image → 3D Model")
	st.write(
		"Upload any RGB photo. The app estimates depth with a transformer model and renders a rough 3-D point cloud."
	)

	st.sidebar.header("3-D Controls")
	max_points = st.sidebar.slider("Point cloud density", 5_000, 60_000, 20_000, step=5_000)
	depth_scale = st.sidebar.slider("Depth exaggeration", 0.5, 3.0, 1.5, step=0.1)
	st.sidebar.caption("Tune the rendering without re-uploading your image.")

	file = st.file_uploader("Drop a JPG/PNG", type=["jpg", "jpeg", "png"])
	if not file:
		st.info("Waiting for an image...")
		return

	image = Image.open(file).convert("RGB")
	img_col, meta_col = st.columns([2, 1])
	with img_col:
		st.image(image, caption="Input image", use_container_width=True)
	with meta_col:
		file_size_kb = getattr(file, "size", len(file.getvalue())) / 1024
		stats_df = pd.DataFrame(
			[
				{"Property": "Resolution", "Value": f"{image.width}×{image.height}"},
				{"Property": "Mode", "Value": image.mode},
				{"Property": "File Size", "Value": f"{file_size_kb:.1f} KB"},
			],
		)
		st.dataframe(stats_df, use_container_width=True, hide_index=True)

	with st.spinner("Running depth estimation..."):
		try:
			depth = estimate_depth_map(image)
		except RuntimeError as exc:
			st.error(str(exc))
			st.stop()
		x, y, z, colors = sample_point_cloud(
			depth,
			image,
			depth_scale=depth_scale,
			max_points=max_points,
		)
		fig = render_point_cloud(x, y, z, colors)
		ply_bytes = point_cloud_to_ply(x, y, z, colors)

	tab_recon, tab_dip = st.tabs(["3-D Reconstruction", "Image Filters Lab"])

	with tab_recon:
		st.subheader("Estimated Depth Map")
		st.image(depth, clamp=True, use_container_width=True)
		st.subheader("Interactive Point Cloud")
		st.plotly_chart(fig, use_container_width=True)
		st.download_button(
			label="Download Point Cloud (PLY)",
			data=ply_bytes,
			file_name="single_image_point_cloud.ply",
			mime="application/octet-stream",
		)

	with tab_dip:
		st.subheader("Digital Image Processing Playground")
		st.caption("Experiment with classic filters before generating the 3-D model.")
		controls_col1, controls_col2, controls_col3 = st.columns(3)
		with controls_col1:
			mean_radius = st.slider("Mean filter radius", 1, 10, 3)
			median_kernel = st.slider("Median kernel size", 3, 15, 5, step=2)
		with controls_col2:
			gaussian_sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.1)
			bilateral_color = st.slider("Bilateral σ_color", 0.01, 0.5, 0.1, step=0.01)
		with controls_col3:
			bilateral_space = st.slider("Bilateral σ_space", 1.0, 10.0, 3.0, step=0.5)
			sharpen_amount = st.slider("Sharpen amount", 0.5, 2.5, 1.2, step=0.1)
		filter_choices = list(FILTER_DESCRIPTIONS.keys())
		selected_filters = st.multiselect(
			"Filters to preview",
			filter_choices,
			default=["Mean", "Median", "Bilateral", "Sharpen"],
		)
		if not selected_filters:
			st.info("Select at least one filter to view the results.")
		else:
			filter_controls = {
				"mean_radius": mean_radius,
				"median_kernel": median_kernel,
				"gaussian_sigma": gaussian_sigma,
				"bilateral_color": bilateral_color,
				"bilateral_space": bilateral_space,
				"sharpen_amount": sharpen_amount,
			}
			filter_results = apply_dip_filters(image, selected_filters, filter_controls)
			if not filter_results:
				st.warning("Unable to compute the requested filters.")
			else:
				summary_df = pd.DataFrame(
					[
						{
							"Filter": res["Filter"],
							"Description": res["Description"],
							"Parameters": res["Parameters"],
						}
						for res in filter_results
					],
				)
				st.dataframe(summary_df, use_container_width=True, hide_index=True)
				preview_cols = st.columns(3)
				for idx, res in enumerate(filter_results):
					with preview_cols[idx % 3]:
						st.image(res["Preview"], caption=res["Filter"], use_container_width=True)


if __name__ == "__main__":
	main()
