"""
VLM-based message handler for Donkey Car simulator.

Receives telemetry (images) from the simulator, runs inference through
a fine-tuned Qwen-VL model, and sends steering commands back.
"""

import base64
import logging
from collections import deque
from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler

from .steering_buckets import token_to_steering

logger = logging.getLogger(__name__)


class VLMDonkeyHandler(IMesgHandler):
    """
    Message handler that uses a fine-tuned VLM for steering predictions.

    Implements the IMesgHandler interface to communicate with the
    Donkey Car simulator over TCP.
    """

    def __init__(
        self,
        model,
        processor,
        constant_throttle: float = 0.3,
        num_buckets: int = 7,
        device: str = "cuda",
        smoothing_window: int = 1,
        prompt: str = "Drive.",
    ):
        """
        Initialize the VLM handler.

        Args:
            model: Fine-tuned Qwen-VL model
            processor: Qwen-VL processor for tokenization
            constant_throttle: Fixed throttle value (0-1)
            num_buckets: Number of steering buckets (5 or 7)
            device: Device to run inference on
            smoothing_window: Number of frames to average for smoothing (1=no smoothing)
            prompt: Text prompt for the model
        """
        self.model = model
        self.processor = processor
        self.constant_throttle = constant_throttle
        self.num_buckets = num_buckets
        self.device = device
        self.prompt = prompt

        # Smoothing buffer
        self.smoothing_window = smoothing_window
        self.steering_history = deque(maxlen=smoothing_window)

        # State
        self.client = None
        self.timer = FPSTimer()

        # Message handlers
        self.fns = {"telemetry": self.on_telemetry}

        # Metrics tracking
        self.frame_count = 0
        self.total_inference_time = 0.0

    def on_connect(self, client) -> None:
        """Called when connected to simulator."""
        self.client = client
        self.timer.reset()
        self.frame_count = 0
        self.total_inference_time = 0.0
        logger.info("Connected to simulator")

    def on_recv_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming message from simulator."""
        self.timer.on_frame()

        if "msg_type" not in message:
            logger.warning(f"Message missing msg_type: {message}")
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)

    def on_telemetry(self, data: Dict[str, Any]) -> None:
        """Process telemetry and send control command."""
        import time

        start_time = time.time()

        # Decode image from base64
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))

        # Get steering prediction from VLM
        steering = self.predict(image)

        # Apply smoothing if enabled
        self.steering_history.append(steering)
        if self.smoothing_window > 1:
            steering = np.mean(self.steering_history)

        # Send control command
        self.send_control(steering, self.constant_throttle)

        # Track metrics
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1

        # Log periodically
        if self.frame_count % 100 == 0:
            avg_time = self.total_inference_time / self.frame_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            logger.info(
                f"Frame {self.frame_count} | "
                f"Avg inference: {avg_time*1000:.1f}ms | "
                f"FPS: {fps:.1f}"
            )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        """
        Predict steering from image using VLM.

        Args:
            image: PIL Image from simulator (120x160 RGB)

        Returns:
            Steering value in range [-1, 1]
        """
        # Prepare input in Qwen-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Generate single token
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,  # Greedy decoding for deterministic driving
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Decode output token
        generated_ids = outputs[0, inputs.input_ids.shape[1] :]
        token = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Convert token to steering value
        steering = token_to_steering(token, self.num_buckets)

        return steering

    def send_control(self, steer: float, throttle: float, brake: float = 0.0) -> None:
        """Send control message to simulator."""
        msg = {
            "msg_type": "control",
            "steering": str(steer),
            "throttle": str(throttle),
            "brake": str(brake),
        }
        self.client.queue_message(msg)

    def on_disconnect(self) -> None:
        """Called when disconnected from simulator."""
        if self.frame_count > 0:
            avg_time = self.total_inference_time / self.frame_count
            logger.info(
                f"Disconnected. Total frames: {self.frame_count}, "
                f"Avg inference: {avg_time*1000:.1f}ms"
            )

    def on_close(self) -> None:
        """Called when handler is closed."""
        pass
