
import os
import io
import base64
import traceback
from types import SimpleNamespace

import torch
import numpy as np
import runpod

from PIL import Image
from torchvision import transforms

from models.networks import define_G


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "/app/checkpoints/best_ckpt.pt"
)

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))

model = None

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    global model

    print("=" * 60)
    print("SNZ CHANGEFORMER WORKER STARTING")
    print("Model: ChangeFormerV6")
    print("Dataset checkpoint: LEVIR-CD")
    print("Device:", DEVICE)
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ChangeFormer requires a CUDA GPU."
        )

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    args = SimpleNamespace(
        net_G="ChangeFormerV6",
        embed_dim=256,
        n_class=2,
        gpu_ids=[0],
    )

    model = define_G(
        args=args,
        gpu_ids=[]
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if "model_G_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint is missing model_G_state_dict. "
            "It may not be the expected ChangeFormer checkpoint."
        )

    model.load_state_dict(
        checkpoint["model_G_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("ChangeFormerV6 loaded successfully.")


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_base64_image(image_b64):
    if not image_b64:
        raise ValueError("Image Base64 data is empty.")

    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]

    image_bytes = base64.b64decode(
        image_b64,
        validate=True
    )

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    tensor = transform(image)
    return tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# MASK ENCODING
# ============================================================

def encode_mask(mask_array):
    buffer = io.BytesIO()

    Image.fromarray(
        mask_array.astype(np.uint8)
    ).save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# CHANGE DETECTION
# ============================================================

@torch.inference_mode()
def predict_change(image_t1, image_t2):
    image_a = preprocess_image(image_t1)
    image_b = preprocess_image(image_t2)

    outputs = model(image_a, image_b)

    # The official evaluator uses the final output.
    logits = outputs[-1]

    prediction = torch.argmax(
        logits,
        dim=1
    )[0]

    # 0 = unchanged, 1 = changed.
    mask = (
        prediction.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return mask


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    job_input = job.get("input", {})

    try:
        image_t1_b64 = (
            job_input.get("image_t1_b64")
            or job_input.get("image1_b64")
        )

        image_t2_b64 = (
            job_input.get("image_t2_b64")
            or job_input.get("image2_b64")
        )

        if not image_t1_b64 or not image_t2_b64:
            raise ValueError(
                "Provide image_t1_b64 and image_t2_b64."
            )

        image_t1 = decode_base64_image(image_t1_b64)
        image_t2 = decode_base64_image(image_t2_b64)

        if image_t1.size != image_t2.size:
            raise ValueError(
                "The before and after images must have "
                "the same dimensions."
            )

        original_width, original_height = image_t1.size

        mask = predict_change(
            image_t1,
            image_t2
        )

        # Resize mask back to original image dimensions.
        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)
        )

        mask_image = mask_image.resize(
            (original_width, original_height),
            Image.Resampling.NEAREST
        )

        mask_array = np.array(
            mask_image,
            dtype=np.uint8
        )

        mask_b64 = encode_mask(mask_array)

        changed_pixels = int(
            np.count_nonzero(mask_array)
        )

        total_pixels = int(
            mask_array.size
        )

        change_percentage = (
            100.0 * changed_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        print(
            "ChangeFormer inference complete:",
            {
                "width": original_width,
                "height": original_height,
                "changed_pixels": changed_pixels,
                "change_percentage": round(
                    change_percentage,
                    2
                ),
            }
        )

        return {
            "success": True,
            "model_used": "ChangeFormerV6-LEVIR-CD",
            "mask_b64": mask_b64,
            "mask_format": "PNG",
            "mask_encoding": "base64",
            "mask_width": original_width,
            "mask_height": original_height,
            "mask_values": {
                "0": "unchanged",
                "255": "changed"
            },
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(
                change_percentage,
                2
            ),
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "success": False,
            "error": str(exc),
            "model_used": "ChangeFormerV6-LEVIR-CD",
        }


# ============================================================
# START WORKER
# ============================================================

if __name__ == "__main__":
    load_model()

    runpod.serverless.start({
        "handler": handler
    })