"""Invisible, user-priority warmup for the baked FLUX Q5 Inductor graph."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid

import execution
import server


BAKED_MODEL = "flux1-dev-Q5_K_S-aidmaNSFWunlock-V0.2-baked-s1.gguf"
TEXT_ENCODER = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
CLIP_MODEL = "clip_l.safetensors"
WARMUP_SOURCE = "baked-flux-startup-warmup"
WARMUP_CLIENT = "baked-flux-startup-warmup-hidden"
IDLE_GRACE_SECONDS = 12.0


class _BakedFluxWarmupDiscard:
    """Internal terminal node: force latent execution without writing a file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "discard"
    OUTPUT_NODE = True
    CATEGORY = "advanced/internal"

    def discard(self, samples):
        return ()


# The internal output is required for prompt validation. It never saves or previews.
NODE_CLASS_MAPPINGS = {
    "_BakedFluxWarmupDiscard": _BakedFluxWarmupDiscard,
}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _warmup_prompt() -> dict:
    # IDs and upstream topology deliberately match FLUX_Baked_Q5_Inductor_768.json.
    # ComfyUI's process-local node cache uses these IDs; changing them prevents the
    # warmed TorchCompileModel output from being reused by the first real Generate.
    return {
        "68": {
            "inputs": {"width": 768, "height": 768, "batch_size": 1},
            "class_type": "EmptySD3LatentImage",
        },
        "70": {
            "inputs": {
                "seed": 513640746507219,
                "steps": 20,
                "cfg": 3.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "beta",
                "denoise": 1.0,
                "model": ["79", 0],
                "positive": ["76", 0],
                "negative": ["77", 0],
                "latent_image": ["68", 0],
            },
            "class_type": "KSampler",
        },
        "72": {
            "inputs": {"unet_name": BAKED_MODEL},
            "class_type": "UnetLoaderGGUF",
        },
        "75": {
            "inputs": {
                "clip_name1": TEXT_ENCODER,
                "clip_name2": CLIP_MODEL,
                "type": "flux",
            },
            "class_type": "DualCLIPLoaderGGUF",
        },
        "76": {
            "inputs": {
                "from_translate": "tr",
                "to_translate": "en",
                "manual_translate": False,
                "Manual Trasnlate": "Manual Trasnlate",
                "text": (
                    "aidmaNSFWunlock, beautiful woman, 25 years old, long blonde hair, "
                    "blue eyes, fit body, wearing lingerie, sitting on a bed, soft natural "
                    "lighting, photorealistic, high detail, sharp focus"
                ),
                "clip": ["75", 0],
            },
            "class_type": "GoogleTranslateCLIPTextEncodeNode",
        },
        "77": {
            "inputs": {
                "from_translate": "tr",
                "to_translate": "en",
                "manual_translate": False,
                "Manual Trasnlate": "Manual Trasnlate",
                "text": (
                    "low quality, blurry, bad anatomy, extra limbs, deformed hands, ugly, "
                    "watermark, text, cartoon"
                ),
                "clip": ["75", 0],
            },
            "class_type": "GoogleTranslateCLIPTextEncodeNode",
        },
        "79": {
            "inputs": {"backend": "inductor", "model": ["72", 0]},
            "class_type": "TorchCompileModel",
        },
        "90": {
            "inputs": {"samples": ["70", 0]},
            "class_type": "_BakedFluxWarmupDiscard",
        },
    }


def _is_warmup_item(item) -> bool:
    try:
        return item[3].get("comfy_usage_source") == WARMUP_SOURCE
    except (AttributeError, IndexError, TypeError):
        return False


def _matches_baked_graph(json_data: dict) -> bool:
    """Recognize a user run that itself makes this warmup redundant."""
    try:
        prompt = json_data["prompt"]
        latent = prompt["68"]["inputs"]
        sampler = prompt["70"]["inputs"]
        loader = prompt["72"]["inputs"]
        compiler = prompt["79"]["inputs"]
        return (
            latent["width"] == 768
            and latent["height"] == 768
            and latent.get("batch_size", 1) == 1
            and sampler["steps"] == 20
            and sampler["sampler_name"] == "dpmpp_2m"
            and sampler["scheduler"] == "beta"
            and sampler["model"] == ["79", 0]
            and loader["unet_name"] == BAKED_MODEL
            and compiler["backend"] == "inductor"
            and compiler["model"] == ["72", 0]
        )
    except (KeyError, TypeError):
        return False


class _WarmupController:
    def __init__(self, prompt_server):
        self.server = prompt_server
        self.queue = prompt_server.prompt_queue
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.prompt_id = None
        self.ready = False
        self.last_user_activity = time.monotonic()

        # Keep references to the real queue views for the worker/controller.
        self._get_tasks_remaining = self.queue.get_tasks_remaining
        self._get_current_queue = self.queue.get_current_queue
        self._get_current_queue_volatile = self.queue.get_current_queue_volatile
        self._delete_queue_item = self.queue.delete_queue_item
        self._interrupt_if_running = self.queue.interrupt_if_running

        # The worker still consumes the real queue, while status broadcasts and
        # /queue omit this maintenance item completely.
        self.queue.get_tasks_remaining = self._visible_tasks_remaining
        self.queue.get_current_queue = self._visible_current_queue
        self.queue.get_current_queue_volatile = self._visible_current_queue_volatile
        self.server.add_on_prompt_handler(self._on_prompt)

    @staticmethod
    def _visible(items):
        return [item for item in items if not _is_warmup_item(item)]

    def _visible_tasks_remaining(self):
        running, pending = self._get_current_queue_volatile()
        return len(self._visible(running)) + len(self._visible(pending))

    def _visible_current_queue(self):
        running, pending = self._get_current_queue()
        return self._visible(running), self._visible(pending)

    def _visible_current_queue_volatile(self):
        running, pending = self._get_current_queue_volatile()
        return self._visible(running), self._visible(pending)

    def _on_prompt(self, json_data: dict) -> dict:
        source = json_data.get("extra_data", {}).get("comfy_usage_source")
        if source == WARMUP_SOURCE:
            return json_data

        with self.lock:
            self.last_user_activity = time.monotonic()
            if _matches_baked_graph(json_data):
                # This real Generate follows the exact target path and will make
                # the process-local graph/cache ready itself.
                self.ready = True
            prompt_id = self.prompt_id
            self.cancel.set()

        if prompt_id is not None:
            self._delete_queue_item(lambda item: item[1] == prompt_id)
            self._interrupt_if_running(prompt_id)
        return json_data

    def _wait_until_idle(self) -> bool:
        while not self.ready:
            if self._get_tasks_remaining() != 0:
                time.sleep(0.5)
                continue
            with self.lock:
                quiet_for = time.monotonic() - self.last_user_activity
            if quiet_for >= IDLE_GRACE_SECONDS:
                return True
            time.sleep(min(0.5, IDLE_GRACE_SECONDS - quiet_for))
        return False

    def _enqueue(self) -> tuple[str, threading.Event]:
        prompt_id = str(uuid.uuid4())
        prompt = _warmup_prompt()
        self.server.node_replace_manager.apply_replacements(prompt)

        future = asyncio.run_coroutine_threadsafe(
            execution.validate_prompt(prompt_id, prompt, None), self.server.loop
        )
        valid = future.result(timeout=90.0)
        if not valid[0]:
            raise RuntimeError(f"warmup prompt validation failed: {valid[1]}")

        cancel = threading.Event()
        with self.lock:
            self.cancel = cancel
            self.prompt_id = prompt_id

        extra_data = {
            "client_id": WARMUP_CLIENT,
            "comfy_usage_source": WARMUP_SOURCE,
            "create_time": int(time.time() * 1000),
        }
        with self.queue.mutex:
            number = -self.server.number
            self.server.number += 1
            self.queue.put((number, prompt_id, prompt, extra_data, valid[2], {}))
        return prompt_id, cancel

    def _wait_result(self, prompt_id: str, cancel: threading.Event):
        deadline = time.monotonic() + 1200.0
        while time.monotonic() < deadline:
            history = self.queue.get_history(prompt_id=prompt_id)
            item = history.get(prompt_id)
            if item is not None:
                return item.get("status")
            if cancel.is_set():
                return None
            time.sleep(0.5)
        raise TimeoutError("warmup timed out")

    def run(self):
        try:
            while not self.server.loop.is_running():
                time.sleep(0.25)

            while not self.ready:
                if not self._wait_until_idle():
                    return

                prompt_id, cancel = self._enqueue()
                status = self._wait_result(prompt_id, cancel)

                # If user work interrupted the maintenance job, wait until that
                # job and the user queue are fully clear before considering retry.
                if cancel.is_set():
                    while self._get_tasks_remaining() != 0:
                        time.sleep(0.5)
                    self.queue.delete_history_item(prompt_id)
                    with self.lock:
                        self.prompt_id = None
                    continue

                self.queue.delete_history_item(prompt_id)
                with self.lock:
                    self.prompt_id = None

                if status and status.get("status_str") == "success":
                    self.ready = True
                    logging.info("[Baked FLUX Warmup] Ready")
                    return
                raise RuntimeError(f"warmup execution failed: {status}")
        except Exception as error:
            # Warmup is optional maintenance; never take the main application down.
            logging.warning("[Baked FLUX Warmup] Failed; ComfyUI continues: %s", error)


def _enabled() -> bool:
    return os.environ.get("COMFY_BAKED_WARMUP", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


if _enabled():
    try:
        _controller = _WarmupController(server.PromptServer.instance)
        threading.Thread(
            target=_controller.run,
            name="baked-flux-inductor-warmup",
            daemon=True,
        ).start()
    except Exception as error:
        logging.warning("[Baked FLUX Warmup] Failed; ComfyUI continues: %s", error)
