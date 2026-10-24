"""Subprocess runner for SAM3 text- and point-grounded segmentation.

Launched by :class:`carryon.recovery.geometry.Sam3Segmenter` using the SAM3
conda env, so CarryOn's main process never imports torch/transformers. It takes
an RGB image plus either a text prompt or pixel ``point_coords`` and writes a
binary target mask (aligned to the input image size) plus structured JSON metadata.

Two modes:

* **one-shot** (default): load the model, segment a single image, exit. Kept as a
  robust fallback.
* **persistent** (``--serve``): load the model **once**, then loop reading one
  JSON request per line on stdin and writing mask/meta files + a one-line JSON
  ack on stdout. This removes the per-call torch-import + ``from_pretrained``
  cost (~6-10 s) that otherwise dominates every ``perceive_grasp``.

Text requests and point requests both use the transformers ``Sam3Model`` stack.
Point prompts are implemented as a small ``input_boxes`` around the pixel
(coordinate prompt), avoiding a second native ``sam3`` checkpoint load.

Output JSON schema (meta file, both modes)::

    {"ok": true, "score": 0.94, "num_masks": 1, "box": [...], "mask_pixels": 998,
     "image_size_hw": [480, 640]}
    {"ok": false, "reason": "no mask for prompt 'carrot'"}
    {"ok": true, "mode": "point", "score": 0.91, ...}

Persistent-mode stdout protocol (one JSON object per line, on the REAL stdout;
all model/inference chatter is redirected away so the stream stays clean)::

    {"ready": true}                       # emitted once the model is loaded
    {"req_ok": true, "meta": "<path>"}    # after each request is written
    {"req_ok": false, "error": "..."}     # on a per-request failure
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

def _to_numpy(x):
    if hasattr(x, 'detach'):
        t = x.detach().cpu()
        if hasattr(t, 'dtype') and t.dtype == getattr(__import__('torch'), 'bfloat16'):
            t = t.float()
        return t.numpy()
    import numpy as np
    return np.asarray(x)

def _point_box_radius_px(image, *, point_box_radius_px: float) -> float:
    r = float(point_box_radius_px)
    if r > 0:
        return r
    return max(8.0, min(24.0, 0.02 * float(min(image.height, image.width))))

def _load_image(np, Image, *, image_npy: str, image: str):
    if image_npy:
        arr = np.load(image_npy)
        return Image.fromarray(np.asarray(arr).astype(np.uint8)).convert('RGB')
    if image:
        return Image.open(image).convert('RGB')
    raise ValueError('either image_npy or image is required')

def _resize_mask_to_image(np, mask, image):
    """Force ``mask`` back onto the image pixel grid (defensive vs model outputs
    at a different resolution). Uses PIL nearest-neighbour rather than cv2 -- the
    SAM3 env's cv2 pulls a libtiff/libjpeg with a broken symbol, and PIL is
    already a hard dependency here."""
    if mask.shape == (image.height, image.width):
        return mask
    from PIL import Image as _Image
    resized = _Image.fromarray(mask.astype(np.uint8) * 255).resize((image.width, image.height), resample=_Image.NEAREST)
    return np.asarray(resized).astype(bool)

def _segment_to_files(*, model, processor, torch, np, image, prompt: str, threshold: float, mask_threshold: float, output: str, meta: str, return_all: bool=False) -> None:
    """Run one segmentation and write the mask (.npy) + meta (.json) files.

    Never raises for a "no mask"/inference issue -- it records a structured
    ``{"ok": false, ...}`` meta instead, matching the one-shot contract.

    With ``return_all`` the mask file is a stacked ``(N, H, W)`` bool array (one
    plane per detected instance, ordered by score desc) and the meta carries
    per-instance ``scores``/``boxes``/``mask_pixels`` lists. This is what
    spatial disambiguation consumes: SAM3 happily enumerates every "can" in the
    scene, and the caller picks the instance nearest the VLA's intent rather
    than betting on the (often wrong) argmax-score mask.
    """
    meta_path = Path(meta)
    with contextlib.redirect_stdout(io.StringIO()):
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(image.height, image.width)])[0]
    masks = results.get('masks')
    scores = results.get('scores')
    boxes = results.get('boxes')
    n = 0 if masks is None else len(masks)
    if n == 0:
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'no mask for prompt {prompt!r}'}), encoding='utf-8')
        return
    scores_np = _to_numpy(scores).reshape(-1)
    order = np.argsort(scores_np)[::-1]
    if return_all:
        planes = []
        out_scores = []
        out_boxes = []
        out_pixels = []
        for idx in order.tolist():
            mk = _resize_mask_to_image(np, _to_numpy(masks[idx]).astype(bool), image)
            planes.append(mk)
            out_scores.append(float(scores_np[idx]))
            out_boxes.append(_to_numpy(boxes[idx]).reshape(-1).tolist() if boxes is not None else None)
            out_pixels.append(int(mk.sum()))
        stacked = np.stack(planes, axis=0)
        np.save(output, stacked)
        meta_path.write_text(json.dumps({'ok': True, 'multi': True, 'num_masks': int(n), 'scores': out_scores, 'boxes': out_boxes, 'mask_pixels': out_pixels, 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')
        return
    top_idx = int(order[0])
    mask = _resize_mask_to_image(np, _to_numpy(masks[top_idx]).astype(bool), image)
    np.save(output, mask)
    box = _to_numpy(boxes[top_idx]).reshape(-1).tolist() if boxes is not None else None
    meta_path.write_text(json.dumps({'ok': True, 'score': float(scores_np[top_idx]), 'num_masks': int(n), 'box': box, 'mask_pixels': int(mask.sum()), 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')

def _segment_point_to_files(*, model, processor, torch, np, image, point_coords, threshold: float, mask_threshold: float, output: str, meta: str, point_box_radius_px: float=12.0) -> None:
    """Run point-prompt segmentation via a small box around ``point_coords``."""
    meta_path = Path(meta)
    try:
        px = float(point_coords[0])
        py = float(point_coords[1])
    except (TypeError, ValueError, IndexError):
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'bad point_coords {point_coords!r}'}), encoding='utf-8')
        return
    r = _point_box_radius_px(image, point_box_radius_px=point_box_radius_px)
    x1 = max(0.0, px - r)
    y1 = max(0.0, py - r)
    x2 = min(float(image.width - 1), px + r)
    y2 = min(float(image.height - 1), py + r)
    box = [[[x1, y1, x2, y2]]]
    labels = [[1]]
    with contextlib.redirect_stdout(io.StringIO()):
        inputs = processor(images=image, text='visual', input_boxes=box, input_boxes_labels=labels, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(outputs, threshold=threshold, mask_threshold=mask_threshold, target_sizes=[(image.height, image.width)])[0]
    masks = results.get('masks')
    scores = results.get('scores')
    n = 0 if masks is None else len(masks)
    if n == 0:
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'no mask at point ({px:.1f},{py:.1f})'}), encoding='utf-8')
        return
    scores_np = _to_numpy(scores).reshape(-1)
    order = np.argsort(scores_np)[::-1]
    top_idx = int(order[0])
    mask = _resize_mask_to_image(np, _to_numpy(masks[top_idx]).astype(bool), image)
    score = float(scores_np[top_idx])
    np.save(output, mask)
    meta_path.write_text(json.dumps({'ok': True, 'mode': 'point', 'score': score, 'num_masks': int(n), 'point_coords': [px, py], 'point_box': [x1, y1, x2, y2], 'mask_pixels': int(mask.sum()), 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')

def _segment_text_native(*, model, processor, torch, np, image, prompt: str, output: str, meta: str, return_all: bool=False, threshold: Optional[float]=None) -> None:
    """Native SAM3 text grounding (cap-x aligned): ``set_image`` +
    ``set_text_prompt``. Masks come back at image resolution, already filtered by
    the processor's load-time ``confidence_threshold`` (kept LOW so the client can
    re-filter per request). ``threshold`` (when given) applies a per-REQUEST score
    gate on top -- this is what lets the caller run an adaptive cascade (a strict
    primary pass, then a relaxed retry) against one persistently-loaded model.
    Writes the same meta/mask files as the transformers path."""
    meta_path = Path(meta)
    with contextlib.redirect_stdout(io.StringIO()):
        state = processor.set_image(image)
        processor.set_text_prompt(prompt, state)
    masks = state.get('masks')
    scores = state.get('scores')
    boxes = state.get('boxes')
    n = 0 if masks is None else int(len(masks))
    if n == 0:
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'no mask for prompt {prompt!r}'}), encoding='utf-8')
        return
    masks_np = _to_numpy(masks)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)
    scores_np = _to_numpy(scores).reshape(-1)
    boxes_np = _to_numpy(boxes) if boxes is not None else None
    order = np.argsort(scores_np)[::-1]
    if threshold is not None:
        order = np.asarray([i for i in order.tolist() if float(scores_np[i]) >= float(threshold)], dtype=int)
        if order.size == 0:
            meta_path.write_text(json.dumps({'ok': False, 'reason': f'no mask >= {float(threshold):.3f} for prompt {prompt!r}'}), encoding='utf-8')
            return
        n = int(order.size)
    if return_all:
        planes, out_scores, out_boxes, out_pixels = ([], [], [], [])
        for idx in order.tolist():
            mk = _resize_mask_to_image(np, masks_np[idx].astype(bool), image)
            planes.append(mk)
            out_scores.append(float(scores_np[idx]))
            out_boxes.append(boxes_np[idx].reshape(-1).tolist() if boxes_np is not None else None)
            out_pixels.append(int(mk.sum()))
        np.save(output, np.stack(planes, axis=0))
        meta_path.write_text(json.dumps({'ok': True, 'multi': True, 'num_masks': int(n), 'scores': out_scores, 'boxes': out_boxes, 'mask_pixels': out_pixels, 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')
        return
    top_idx = int(order[0])
    mask = _resize_mask_to_image(np, masks_np[top_idx].astype(bool), image)
    np.save(output, mask)
    box = boxes_np[top_idx].reshape(-1).tolist() if boxes_np is not None else None
    meta_path.write_text(json.dumps({'ok': True, 'score': float(scores_np[top_idx]), 'num_masks': int(n), 'box': box, 'mask_pixels': int(mask.sum()), 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')

def _segment_point_native(*, model, processor, torch, np, image, point_coords, output: str, meta: str) -> None:
    """Native SAM3 promptable point segmentation (cap-x aligned): a single
    foreground point -> full object mask via ``predict_inst`` with
    ``multimask_output`` (pick the top-IoU mask). This replaces the 12px
    ``input_boxes`` approximation that clamped the mask to a table-height sliver."""
    meta_path = Path(meta)
    try:
        px = float(point_coords[0])
        py = float(point_coords[1])
    except (TypeError, ValueError, IndexError):
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'bad point_coords {point_coords!r}'}), encoding='utf-8')
        return
    with contextlib.redirect_stdout(io.StringIO()):
        state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(state, point_coords=np.array([[px, py]], dtype=np.float32), point_labels=np.array([1], dtype=np.int64), multimask_output=True)
    masks_np = _to_numpy(masks)
    scores_np = _to_numpy(scores).reshape(-1)
    if masks_np.size == 0 or scores_np.size == 0:
        meta_path.write_text(json.dumps({'ok': False, 'reason': f'no mask at point ({px:.1f},{py:.1f})'}), encoding='utf-8')
        return
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    order = np.argsort(scores_np)[::-1]
    point_x = min(max(int(round(px)), 0), int(image.width) - 1)
    point_y = min(max(int(round(py)), 0), int(image.height) - 1)
    resized = {int(index): _resize_mask_to_image(np, masks_np[int(index)].astype(bool), image) for index in order.tolist()}
    containing = [int(index) for index in order.tolist() if bool(resized[int(index)][point_y, point_x])]
    top = containing[0] if containing else int(order[0])
    raw_mask_files = []
    raw_dir = os.environ.get('CARRYON_SAM3_RAW_MASK_DIR', '').strip()
    if raw_dir:
        from PIL import Image as _Image
        raw_root = Path(raw_dir).resolve()
        raw_root.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(order.tolist()):
            raw_mask = resized[int(index)]
            stem = f'point_{point_x:04d}_{point_y:04d}_rank{rank:02d}'
            npy_path = raw_root / f'{stem}.npy'
            png_path = raw_root / f'{stem}.png'
            np.save(npy_path, raw_mask)
            _Image.fromarray(raw_mask.astype(np.uint8) * 255).save(png_path)
            raw_mask_files.append({'rank': int(rank), 'model_index': int(index), 'score': float(scores_np[int(index)]), 'mask_pixels': int(raw_mask.sum()), 'npy': str(npy_path), 'png': str(png_path)})
    mask = resized[top]
    np.save(output, mask)
    meta_path.write_text(json.dumps({'ok': True, 'mode': 'point', 'score': float(scores_np[top]), 'num_masks': int(masks_np.shape[0]), 'point_coords': [px, py], 'selected_mask_index': int(top), 'point_contained': bool(mask[point_y, point_x]), 'candidate_scores': [float(scores_np[int(i)]) for i in order.tolist()], 'raw_masks': raw_mask_files, 'mask_pixels': int(mask.sum()), 'image_size_hw': [int(image.height), int(image.width)]}), encoding='utf-8')

def _load_native(args, torch, device: str):
    """Load native SAM3 (``sam3`` package) with the instance-interactive
    predictor enabled, from a local checkpoint (offline)."""
    import os
    from sam3.model.sam3_image_processor import Sam3Processor as _NativeProcessor
    from sam3.model_builder import build_sam3_image_model
    ckpt = args.model
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, 'sam3.pt')
    dev = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False, enable_inst_interactivity=True, device=dev)
        if hasattr(model, 'to'):
            model = model.to(dev)
        processor = _NativeProcessor(model, device=dev, confidence_threshold=float(args.threshold))
    return (model, processor)

def _resolve_device(torch, dev: str) -> str:
    dev = (dev or 'auto').strip().lower()
    if dev == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev in ('cuda', 'cpu'):
        return dev
    raise ValueError(f'unsupported --device {dev!r} (use auto, cuda, or cpu)')

def _load_model(args):
    import numpy as np
    import torch
    from PIL import Image
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(torch, getattr(args, 'device', None))
    backend = getattr(args, 'backend', 'native')
    if backend == 'native':
        model, processor = _load_native(args, torch, device)
    else:
        from transformers import Sam3Model, Sam3Processor
        with contextlib.redirect_stdout(io.StringIO()):
            model = Sam3Model.from_pretrained(args.model).to(device)
            processor = Sam3Processor.from_pretrained(args.model)
            model.eval()
    return (backend, model, processor, torch, np, Image)

def _serve(args) -> int:
    """Persistent mode: load once, then segment one stdin request per line.

    Protocol lines go to the *real* stdout; model/inference prints are redirected
    inside the helpers so they never corrupt the stream.
    """
    real_stdout = sys.stdout

    def _emit(obj: dict) -> None:
        real_stdout.write(json.dumps(obj) + '\n')
        real_stdout.flush()
    try:
        backend, model, processor, torch, np, Image = _load_model(args)
    except Exception as exc:
        _emit({'ready': False, 'error': f'{type(exc).__name__}: {exc}'})
        return 1
    _emit({'ready': True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            _emit({'req_ok': False, 'error': f'bad request json: {exc}'})
            continue
        if req.get('cmd') == 'shutdown':
            break
        try:
            image = _load_image(np, Image, image_npy=req.get('image_npy', ''), image=req.get('image', ''))
            if req.get('point_coords') is not None:
                if backend == 'native':
                    _segment_point_native(model=model, processor=processor, torch=torch, np=np, image=image, point_coords=req['point_coords'], output=str(req['output']), meta=str(req['meta']))
                else:
                    _segment_point_to_files(model=model, processor=processor, torch=torch, np=np, image=image, point_coords=req['point_coords'], threshold=float(req.get('threshold', args.threshold)), mask_threshold=float(req.get('mask_threshold', args.mask_threshold)), output=str(req['output']), meta=str(req['meta']), point_box_radius_px=float(req.get('point_box_radius_px', 12.0)))
            elif backend == 'native':
                _segment_text_native(model=model, processor=processor, torch=torch, np=np, image=image, prompt=str(req['prompt']), output=str(req['output']), meta=str(req['meta']), return_all=bool(req.get('all_masks', False)), threshold=float(req.get('threshold', args.threshold)))
            else:
                _segment_to_files(model=model, processor=processor, torch=torch, np=np, image=image, prompt=str(req['prompt']), threshold=float(req.get('threshold', args.threshold)), mask_threshold=float(req.get('mask_threshold', args.mask_threshold)), output=str(req['output']), meta=str(req['meta']), return_all=bool(req.get('all_masks', False)))
            _emit({'req_ok': True, 'meta': str(req['meta'])})
        except Exception as exc:
            with contextlib.suppress(Exception):
                Path(req['meta']).write_text(json.dumps({'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}), encoding='utf-8')
            _emit({'req_ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--serve', action='store_true', help='persistent mode: load once, segment stdin requests')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--point-x', type=float, default=None, help='pixel x for point-prompt mode')
    parser.add_argument('--point-y', type=float, default=None, help='pixel y for point-prompt mode')
    parser.add_argument('--image', default='')
    parser.add_argument('--image-npy', default='')
    parser.add_argument('--output', default='', help='mask .npy path (bool HxW)')
    parser.add_argument('--meta', default='', help='result .json path')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--all-masks', action='store_true', help='write a stacked (N,H,W) mask + per-instance meta')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'), help='inference device (cpu avoids GPU OOM when cuRobo holds VRAM)')
    parser.add_argument('--backend', default='native', choices=('native', 'transformers'), help='native = sam3 package (promptable point -> full object; cap-x aligned); transformers = HF Sam3Model (point approximated by a small box).')
    args = parser.parse_args()
    if args.serve:
        return _serve(args)
    meta_path = Path(args.meta)
    try:
        if not args.meta or not args.output:
            raise ValueError('one-shot mode requires --output and --meta')
        image = None
        if args.point_x is not None and args.point_y is not None:
            backend, model, processor, torch, np, Image = _load_model(args)
            image = _load_image(np, Image, image_npy=args.image_npy, image=args.image)
            if backend == 'native':
                _segment_point_native(model=model, processor=processor, torch=torch, np=np, image=image, point_coords=[args.point_x, args.point_y], output=args.output, meta=args.meta)
            else:
                _segment_point_to_files(model=model, processor=processor, torch=torch, np=np, image=image, point_coords=[args.point_x, args.point_y], threshold=args.threshold, mask_threshold=args.mask_threshold, output=args.output, meta=args.meta)
        else:
            if not args.prompt:
                raise ValueError('one-shot text mode requires --prompt')
            backend, model, processor, torch, np, Image = _load_model(args)
            image = _load_image(np, Image, image_npy=args.image_npy, image=args.image)
            if backend == 'native':
                _segment_text_native(model=model, processor=processor, torch=torch, np=np, image=image, prompt=args.prompt, output=args.output, meta=args.meta, return_all=args.all_masks, threshold=args.threshold)
            else:
                _segment_to_files(model=model, processor=processor, torch=torch, np=np, image=image, prompt=args.prompt, threshold=args.threshold, mask_threshold=args.mask_threshold, output=args.output, meta=args.meta, return_all=args.all_masks)
        return 0
    except Exception as exc:
        with contextlib.suppress(Exception):
            meta_path.write_text(json.dumps({'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}), encoding='utf-8')
        return 0
if __name__ == '__main__':
    raise SystemExit(main())
