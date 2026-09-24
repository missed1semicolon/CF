import os
import io
import base64
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import runpod
from PIL import Image, ImageOps
from torchvision import transforms
from models.networks import define_G


# ============================================================
# SNZ CHANGEFORMER V8
# ============================================================
# ChangeFormer remains a specialist binary change detector. It does NOT try
# to answer object-level questions. The SNZ gateway consumes this mask and
# can automatically invoke RSCoVLM for interpretation/grounding/counting.
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/best_ckpt.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))

model = None

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_model() -> None:
    global model
    print("=" * 70)
    print("SNZ CHANGEFORMER V8 WORKER STARTING")
    print("Model: ChangeFormerV6 / LEVIR-CD checkpoint")
    print("Device:", DEVICE)
    if not torch.cuda.is_available():
        raise RuntimeError("ChangeFormer requires a CUDA GPU.")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    args = SimpleNamespace(net_G="ChangeFormerV6", embed_dim=256, n_class=2, gpu_ids=[0])
    model = define_G(args=args, gpu_ids=[])
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "model_G_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint is missing model_G_state_dict.")
    model.load_state_dict(checkpoint["model_G_state_dict"], strict=True)
    model.to(DEVICE).eval()
    print("ChangeFormer V8 loaded successfully.")


def decode_base64_image(value: str) -> Image.Image:
    if not value:
        raise ValueError("Image Base64 data is empty.")
    value = value.strip()
    if value.startswith("data:image"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(DEVICE)


def encode_mask(mask_array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask_array.astype(np.uint8), mode="L").save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@torch.inference_mode()
def predict_change(image_t1: Image.Image, image_t2: Image.Image) -> np.ndarray:
    a = preprocess_image(image_t1)
    b = preprocess_image(image_t2)
    outputs = model(a, b)
    logits = outputs[-1]
    prediction = torch.argmax(logits, dim=1)[0]
    return prediction.detach().cpu().numpy().astype(np.uint8)


def _changed_bbox(mask: np.ndarray) -> Optional[list]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})
    try:
        image_t1_b64 = job_input.get("image_t1_b64") or job_input.get("image1_b64")
        image_t2_b64 = job_input.get("image_t2_b64") or job_input.get("image2_b64")
        if not image_t1_b64 or not image_t2_b64:
            raise ValueError("Provide image_t1_b64 and image_t2_b64.")

        image_t1 = decode_base64_image(image_t1_b64)
        image_t2 = decode_base64_image(image_t2_b64)
        if image_t1.size != image_t2.size:
            raise ValueError(f"Before/after dimensions differ: {image_t1.size} vs {image_t2.size}.")

        width, height = image_t1.size
        mask = predict_change(image_t1, image_t2)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((width, height), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        mask_b64 = encode_mask(mask_array)

        changed_pixels = int(np.count_nonzero(mask_array))
        total_pixels = int(mask_array.size)
        change_percentage = 100.0 * changed_pixels / total_pixels if total_pixels else 0.0
        bbox = _changed_bbox(mask_array)

        metadata = {
            "model": "ChangeFormerV6-LEVIR-CD",
            "input_dimensions": {"width": width, "height": height},
            "mask_dimensions": {"width": width, "height": height},
            "mask_values": {"0": "unchanged", "255": "changed"},
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 4),
            "changed_region_bbox_xyxy": bbox,
            "interpretation_note": (
                "The mask identifies predicted changed pixels. It does not by itself establish "
                "object identity, destruction, construction, cause, or exact object counts."
            ),
            "analysis_plan": job_input.get("metadata", {}).get("analysis_plan", {}),
        }

        print("ChangeFormer V8 complete:", metadata)

        return {
            "success": True,
            "model_used": "ChangeFormerV6-LEVIR-CD",
            "mask_b64": mask_b64,
            "mask_format": "PNG",
            "mask_encoding": "base64",
            "mask_width": width,
            "mask_height": height,
            "width": width,
            "height": height,
            "mask_values": metadata["mask_values"],
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 4),
            "changed_region_bbox_xyxy": bbox,
            "metadata": metadata,
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "success": False,
            "model_used": "ChangeFormerV6-LEVIR-CD",
            "error": str(exc),
        }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
