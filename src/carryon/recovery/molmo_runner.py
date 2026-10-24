"""Subprocess runner for Molmo2 open-vocabulary *pointing*.

Launched by :class:`carryon.recovery.geometry.MolmoPointer` in the Molmo conda
env (``molmoact`` python + a ``transformers==4.57.1`` overlay on ``PYTHONPATH``),
so CarryOn's main process never imports torch/transformers. Given an RGB image
and one or more object phrases, it asks Molmo2 to point at each and returns the
pixel coordinate of the top point per phrase.

This is the *live* grounding point source: unlike the sim's ground-truth
``target_px`` (the object's 3D centre projected to pixels, which lands below the
visible body under an oblique camera), Molmo points at the visible surface, so a
SAM3 point-prompt around it segments the body -- not a table-height sliver.

Persistent-mode stdout protocol (one JSON object per line on the REAL stdout;
all model/inference chatter is redirected away so the stream stays clean)::

    {"ready": true}                                    # model loaded
    {"req_ok": true, "points": {"ketchup": [270, 279]}}  # px per phrase (or null)
    {"req_ok": false, "error": "..."}                  # per-request failure
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys


# Molmo2: <points coords="type idx x y idx x y ...">label</points>  (0-1000)
_COORDS_RE = re.compile(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
# Molmo1: <point x=".." y=".."> (0-100)
_POINT_RE = re.compile(r"<point\b[^>]*>", re.IGNORECASE)
_XY_RE = re.compile(r"\b([xy])\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", re.IGNORECASE)


def _parse_first_point(text: str) -> tuple[float, float, float] | None:
    """Return (x, y, norm_scale) for the first parsed point, or None.

    norm_scale is 1000.0 for Molmo2 coords and 100.0 for Molmo1 tags.
    """
    m = _COORDS_RE.search(text or "")
    if m:
        nums = [float(n) for n in m.group(1).split()]
        # layout: <type> then triplets (idx, x, y). Take the first (x, y).
        if len(nums) >= 4:
            return nums[2], nums[3], 1000.0
        # Defensive: some outputs omit the type indicator -> (idx, x, y).
        if len(nums) == 3:
            return nums[1], nums[2], 1000.0
    for tag in _POINT_RE.findall(text or ""):
        vals = {k.lower(): float(v) for k, v in _XY_RE.findall(tag)}
        if "x" in vals and "y" in vals:
            return vals["x"], vals["y"], 100.0
    return None


def _point_object(model, processor, torch, image, obj: str, *, max_new_tokens: int):
    prompt = f"Point to the {obj}."
    messages = [{"role": "user", "content": [
        dict(type="text", text=prompt),
        dict(type="image", image=image),
    ]}]
    with contextlib.redirect_stdout(io.StringIO()):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                 do_sample=False)
        out_ids = gen[0, inputs["input_ids"].size(1):]
        text = processor.tokenizer.decode(out_ids, skip_special_tokens=True)
    parsed = _parse_first_point(text)
    if parsed is None:
        return None, text
    x, y, scale = parsed
    W, H = image.size
    px = int(round(x / scale * W))
    py = int(round(y / scale * H))
    px = min(W - 1, max(0, px))
    py = min(H - 1, max(0, py))
    return (px, py), text


def _load_image(np, Image, *, image_npy: str, image: str):
    if image_npy:
        arr = np.load(image_npy)
        return Image.fromarray(np.asarray(arr).astype(np.uint8)).convert("RGB")
    if image:
        return Image.open(image).convert("RGB")
    raise ValueError("either image_npy or image is required")


def _load_model(args):
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True, dtype="auto", device_map="auto"
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, trust_remote_code=True, dtype="auto", device_map="auto"
        )
        model.eval()
    return model, processor, torch, np, Image


def _serve(args) -> int:
    real_stdout = sys.stdout

    def _emit(obj: dict) -> None:
        real_stdout.write(json.dumps(obj) + "\n")
        real_stdout.flush()

    try:
        model, processor, torch, np, Image = _load_model(args)
    except Exception as exc:  # noqa: BLE001
        _emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            _emit({"req_ok": False, "error": f"bad request json: {exc}"})
            continue
        if req.get("cmd") == "shutdown":
            break
        try:
            image = _load_image(
                np, Image,
                image_npy=req.get("image_npy", ""),
                image=req.get("image", ""),
            )
            objects = req.get("objects") or []
            if isinstance(objects, str):
                objects = [objects]
            points: dict[str, list | None] = {}
            raw_outputs: dict[str, str] = {}
            for obj in objects:
                pt, _text = _point_object(
                    model, processor, torch, image, str(obj),
                    max_new_tokens=int(req.get("max_new_tokens", args.max_new_tokens)),
                )
                points[str(obj)] = [int(pt[0]), int(pt[1])] if pt is not None else None
                raw_outputs[str(obj)] = _text
            _emit({"req_ok": True, "request_id": req.get("request_id"),
                   "points": points, "raw_outputs": raw_outputs})
        except Exception as exc:  # noqa: BLE001
            _emit({"req_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.serve:
        return _serve(args)
    raise SystemExit("molmo_runner currently supports --serve only")


if __name__ == "__main__":
    raise SystemExit(main())
