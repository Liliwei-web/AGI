from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any

from loguru import logger
from PIL import Image

from arenaagent.vlm_agent.skills.raven import crop_group_image_to_subplots, solve_raven


def _fail(agent: Any, error: str) -> dict[str, Any]:
    fail_result = getattr(agent, "_fail_result", None)
    if callable(fail_result):
        return fail_result(error=error)
    return {"result": "failed", "error": str(error)}


def run_raven_inference(image_list: list[list[Image.Image]], structure: list[Any]) -> list[list[int]] | None:
    """Run the Raven model and normalize its predictions."""
    argv_backup = sys.argv[:]
    try:
        # solve_raven internally calls argparse.parse_args(), isolate from process args.
        sys.argv = [argv_backup[0]] if argv_backup else [""]
        prediction = solve_raven(image_list=image_list, structure=structure)
    except Exception as exc:
        logger.warning("solve_raven api call failed: {}", exc)
        return None
    finally:
        sys.argv = argv_backup

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    if not isinstance(prediction, list) or not prediction:
        return None

    normalized: list[list[int]] = []
    for triple in prediction:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        try:
            normalized.append([int(triple[0]), int(triple[1]), int(triple[2])])
        except Exception:
            continue
    return normalized or None


def get_raven_ranked_candidates(
    agent: Any,
    cache_key: str,
    image_list: list[list[Image.Image]],
    structure: list[Any],
) -> list[list[int]] | None:
    cache = getattr(agent, "_raven_candidates_cache", None)
    next_index = getattr(agent, "_raven_next_index", None)
    if not isinstance(cache, dict) or not isinstance(next_index, dict):
        return None

    if cache_key in cache:
        return cache[cache_key]

    candidates = run_raven_inference(image_list=image_list, structure=structure)
    if not candidates:
        return None

    cache[cache_key] = candidates
    next_index.setdefault(cache_key, 0)
    return candidates


def normalize_raven_image_list(raw_image_list: Any) -> list[list[Image.Image]] | None:
    # New input format: one whole Raven canvas image (base64/path/bytes/PIL).
    if isinstance(raw_image_list, (str, bytes, Image.Image)):
        return group_image_to_raven_list(raw_image_list)

    if not isinstance(raw_image_list, list):
        return None

    if len(raw_image_list) == 1 and isinstance(raw_image_list[0], (str, bytes, Image.Image)):
        return group_image_to_raven_list(raw_image_list[0])

    # A: [[16 imgs], [16 imgs], [16 imgs]]
    if raw_image_list and isinstance(raw_image_list[0], list):
        groups: list[list[Image.Image]] = []
        for group in raw_image_list:
            pil_group = to_pil_group(group)
            if not pil_group:
                return None
            groups.append(pil_group)
        return groups if groups else None

    pil_images = to_pil_group(raw_image_list)
    if not pil_images:
        return None

    # C: [48 imgs] -> split into 3 groups
    if len(pil_images) >= 48:
        return [pil_images[0:16], pil_images[16:32], pil_images[32:48]]

    return None


def group_image_to_raven_list(group_image: Any) -> list[list[Image.Image]] | None:
    image_path = materialize_group_image(group_image)
    if not image_path:
        return None

    try:
        crop_group_image_to_subplots(image_path)
    except Exception as exc:
        logger.warning("crop_group_image_to_subplots failed: {}", exc)
        return None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    subfolder = os.path.join(os.path.dirname(image_path), base_name)
    groups: list[list[Image.Image]] = []

    try:
        for q_idx in range(1, 4):
            group: list[Image.Image] = []
            for n in range(1, 9):
                problem_path = os.path.join(subfolder, f"question_{q_idx}_problem_0{n}.png")
                if not os.path.exists(problem_path):
                    logger.warning("Missing cropped panel: {}", problem_path)
                    return None
                group.append(Image.open(problem_path).convert("L"))

            for n in range(1, 9):
                answer_path = os.path.join(subfolder, f"question_{q_idx}_answer_0{n}.png")
                if not os.path.exists(answer_path):
                    logger.warning("Missing cropped panel: {}", answer_path)
                    return None
                group.append(Image.open(answer_path).convert("L"))
            groups.append(group)
    except Exception as exc:
        logger.warning("Failed to load cropped Raven panels: {}", exc)
        return None

    return groups if len(groups) == 3 else None


def materialize_group_image(value: Any) -> str | None:
    out_dir = "/tmp/raven_input_images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"group_{int(time.time() * 1000)}_{os.getpid()}.png")

    try:
        if isinstance(value, Image.Image):
            value.save(out_path)
            return out_path

        if isinstance(value, bytes):
            img = Image.open(io.BytesIO(value)).convert("L")
            img.save(out_path)
            return out_path

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            if text.startswith("data:image"):
                _, b64_data = text.split(",", 1)
                raw = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(raw)).convert("L")
                img.save(out_path)
                return out_path

            if os.path.exists(text):
                return text

            # Fallback: treat as raw base64 without data URL prefix.
            raw = base64.b64decode(text)
            img = Image.open(io.BytesIO(raw)).convert("L")
            img.save(out_path)
            return out_path
    except Exception as exc:
        logger.warning("Failed to materialize Raven group image: {}", exc)
        return None

    return None


def to_pil_group(values: Any) -> list[Image.Image] | None:
    if not isinstance(values, list):
        return None
    images: list[Image.Image] = []
    for value in values:
        image = to_pil_image(value)
        if image is None:
            return None
        images.append(image)
    return images


def to_pil_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("L")

    if isinstance(value, bytes):
        try:
            return Image.open(io.BytesIO(value)).convert("L")
        except Exception:
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.startswith("data:image"):
            try:
                _, b64_data = text.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("L")
            except Exception:
                return None

        if os.path.exists(text):
            try:
                return Image.open(text).convert("L")
            except Exception:
                return None

    return None


def build_raven_attempt_key(image_list: list[list[Image.Image]], structure: list[Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(structure, ensure_ascii=False, default=str).encode("utf-8"))
    for group in image_list:
        for image in group:
            hasher.update(image.mode.encode("utf-8"))
            hasher.update(str(image.size).encode("utf-8"))
            hasher.update(image.tobytes())
    return hasher.hexdigest()


def materialize_task_data_images(agent: Any, task_data: Any) -> None:
    cleanup_raven_temp_images(agent)

    created_paths: list[str] = []
    temp_dir: str | None = None
    image_index = 0

    def ensure_temp_dir() -> str:
        nonlocal temp_dir
        if temp_dir is None:
            base_dir = os.path.join(getattr(getattr(agent, "cfg", None), "log_dir", "") or "logs", "raven_task_data")
            os.makedirs(base_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix=f"{getattr(agent, 'agent_id', 'agent')}_", dir=base_dir)
        return temp_dir

    def convert(value: Any) -> Any:
        nonlocal image_index

        if isinstance(value, dict):
            for item in value.values():
                convert(item)
            return value

        if isinstance(value, list):
            for item in value:
                convert(item)
            return value

        if not isinstance(value, str):
            return value

        image_bytes, extension = decode_base64_image(value)
        if image_bytes is None:
            return value

        image_index += 1
        file_path = os.path.join(ensure_temp_dir(), f"task_data_{image_index:03d}{extension}")
        with open(file_path, "wb") as writer:
            writer.write(image_bytes)
        logger.debug("raven file path {}", file_path)
        created_paths.append(file_path)
        return file_path

    convert(task_data)
    if not created_paths:
        logger.debug("no raven image path extracted from task data")
        return

    agent._raven_image_temp_path = created_paths[0] if len(created_paths) == 1 else (temp_dir or created_paths[0])


def cleanup_raven_temp_images(agent: Any) -> None:
    temp_path = getattr(agent, "_raven_image_temp_path", "")
    if not temp_path:
        return

    agent._raven_image_temp_path = ""
    try:
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
            return
        if os.path.isfile(temp_path):
            parent_dir = os.path.dirname(temp_path)
            os.remove(temp_path)
            if parent_dir and os.path.isdir(parent_dir):
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass
    except Exception as exc:
        logger.warning("Failed to cleanup Raven temp images {}: {}", temp_path, exc)


def decode_base64_image(value: str) -> tuple[bytes | None, str]:
    text = value.strip()
    if not text or os.path.exists(text):
        return None, ""

    header = ""
    payload = text
    if text.startswith("data:image"):
        parts = text.split(",", 1)
        if len(parts) != 2:
            return None, ""
        header, payload = parts
    elif len(text) < 128:
        return None, ""

    try:
        image_bytes = base64.b64decode(payload, validate=not bool(header))
    except (binascii.Error, ValueError):
        return None, ""

    extension = guess_image_extension(image_bytes, header)
    if not extension:
        return None, ""
    return image_bytes, extension


def guess_image_extension(image_bytes: bytes, header: str = "") -> str:
    lower_header = header.lower()
    if "image/png" in lower_header:
        return ".png"
    if "image/jpeg" in lower_header or "image/jpg" in lower_header:
        return ".jpg"
    if "image/webp" in lower_header:
        return ".webp"
    if "image/gif" in lower_header:
        return ".gif"
    if "image/bmp" in lower_header:
        return ".bmp"
    if "image/svg+xml" in lower_header:
        return ".svg"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.lstrip().startswith(b"<svg"):
        return ".svg"
    return ""


def resolve_raven_image_path(image_temp_path: str) -> str | None:
    if not image_temp_path:
        return None
    if os.path.isfile(image_temp_path):
        return image_temp_path
    if not os.path.isdir(image_temp_path):
        return None

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    for root, _, files in os.walk(image_temp_path):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            if os.path.splitext(file_name)[1].lower() in image_extensions:
                return file_path
    return None


def handle(agent: Any, params: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Handle the solve_raven action for agents with prepared Raven task data."""
    image_temp_path = getattr(agent, "_raven_image_temp_path", "")
    resolved_image_path = resolve_raven_image_path(image_temp_path)
    if not resolved_image_path:
        logger.warning("raven image path is not exist")
        return _fail(agent, "raven image path not exist")

    get_param = getattr(agent, "_get_param", None)
    if not callable(get_param):
        return _fail(agent, "raven skill helper missing: _get_param")

    structure = get_param(params, "structure", default=[])
    if not isinstance(structure, list):
        structure = []

    logger.debug("solve_raven using image path {}", resolved_image_path)
    image_list = normalize_raven_image_list(resolved_image_path)
    if not image_list:
        logger.warning("invalid parameter: image_list")
        return _fail(agent, "invalid parameter: image_list")

    cache_key = build_raven_attempt_key(image_list, structure)
    ranked_candidates = get_raven_ranked_candidates(agent, cache_key, image_list, structure)
    if not ranked_candidates:
        logger.warning("solve_raven failed")
        return _fail(agent, "solve_raven failed")

    raven_next_index = getattr(agent, "_raven_next_index", None)
    if not isinstance(raven_next_index, dict):
        return _fail(agent, "raven skill helper missing: _raven_next_index")

    attempt_index = raven_next_index.get(cache_key, 0)
    used_index = min(attempt_index, len(ranked_candidates) - 1)
    chosen_answer = ranked_candidates[used_index]

    if attempt_index < len(ranked_candidates) - 1:
        raven_next_index[cache_key] = attempt_index + 1

    logger.debug("raven solved answer {}", chosen_answer)
    action_space = getattr(agent, "action_space", {}) or {}
    key = action_space.get("key") or "action"
    return {key: chosen_answer}
