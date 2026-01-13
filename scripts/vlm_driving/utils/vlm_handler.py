"""
VLM-based message handler for Donkey Car simulator with obstacle avoidance.

Receives telemetry (images) from the simulator, runs inference through
a fine-tuned Qwen-VL model, and sends steering commands back.

Supports turn commands (A/G tokens) for 80-degree obstacle avoidance turns
using a step-count based state machine.
"""

import base64
import logging
from collections import deque
from enum import Enum
from io import BytesIO
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler

from .steering_buckets import ACTION_TOKENS, token_to_steering

logger = logging.getLogger(__name__)


class DrivingState(Enum):
    """State machine for driving with obstacle avoidance."""

    NORMAL = "normal"
    TURNING = "turning"


class VLMDonkeyHandler(IMesgHandler):
    """
    Message handler that uses a fine-tuned VLM for steering predictions
    with obstacle avoidance via turn commands.

    Implements the IMesgHandler interface to communicate with the
    Donkey Car simulator over TCP.
    """

    # Calibrated for 80-degree turn at full steering
    # Run calibrate_turn.py to determine the correct value for your setup
    STEPS_FOR_80_DEGREE_TURN = 100

    def __init__(
        self,
        model,
        processor,
        constant_throttle: float = 0.5,
        turn_throttle: float = 0.2,
        num_buckets: int = 7,
        device: str = "cuda",
        smoothing_window: int = 1,
        prompt: str = "Drive.",
        turn_steps: int = None,
    ):
        """
        Initialize the VLM handler with obstacle avoidance support.

        Args:
            model: Fine-tuned Qwen-VL model
            processor: Qwen-VL processor for tokenization
            constant_throttle: Fixed throttle value for normal driving (0-1)
            turn_throttle: Throttle value during turn maneuvers (0-1)
            num_buckets: Number of steering buckets (5 or 7)
            device: Device to run inference on
            smoothing_window: Number of frames to average for smoothing (1=no smoothing)
            prompt: Text prompt for the model
            turn_steps: Override for STEPS_FOR_80_DEGREE_TURN (optional)
        """
        self.model = model
        self.processor = processor
        self.constant_throttle = constant_throttle
        self.turn_throttle = turn_throttle
        self.num_buckets = num_buckets
        self.device = device
        self.prompt = prompt

        # Turn configuration
        if turn_steps is not None:
            self.turn_steps = turn_steps
        else:
            self.turn_steps = self.STEPS_FOR_80_DEGREE_TURN

        # Smoothing buffer
        self.smoothing_window = smoothing_window
        self.steering_history = deque(maxlen=smoothing_window)

        # State machine for turns
        self.state = DrivingState.NORMAL
        self.turn_steps_remaining = 0
        self.turn_direction = 0.0  # -1.0 for left, +1.0 for right

        # State
        self.client = None
        self.timer = FPSTimer()

        # Message handlers
        self.fns = {"telemetry": self.on_telemetry}

        # Metrics tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.turn_count = 0

    def on_connect(self, client) -> None:
        """Called when connected to simulator."""
        self.client = client
        self.timer.reset()
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.turn_count = 0
        self.state = DrivingState.NORMAL
        self.turn_steps_remaining = 0
        self.steering_history.clear()
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

        # State machine logic
        if self.state == DrivingState.TURNING:
            # Continue executing turn
            steering, throttle = self._continue_turn()
        else:
            # Get prediction from VLM
            token = self.predict_token(image)
            steering, throttle = self._process_token(token)

        # Send control command
        self.send_control(steering, throttle)

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
                f"State: {self.state.value} | "
                f"Turns: {self.turn_count} | "
                f"Avg inference: {avg_time*1000:.1f}ms | "
                f"FPS: {fps:.1f}"
            )

    def _process_token(self, token: str) -> tuple:
        """
        Process VLM output token and return steering/throttle.

        Args:
            token: Predicted token (A-G)

        Returns:
            Tuple of (steering, throttle)
        """
        action_info = ACTION_TOKENS.get(token, ACTION_TOKENS["D"])

        if action_info["type"] == "turn":
            # Initiate turn maneuver
            self._start_turn(action_info["degrees"])
            steering = self.turn_direction
            throttle = self.turn_throttle
        else:
            # Normal steering
            steering = action_info["value"]

            # Apply smoothing
            self.steering_history.append(steering)
            if self.smoothing_window > 1:
                steering = float(np.mean(self.steering_history))

            throttle = self.constant_throttle

        return steering, throttle

    def _start_turn(self, degrees: float) -> None:
        """
        Start a turn maneuver.

        Args:
            degrees: Turn angle (negative = left, positive = right)
        """
        self.state = DrivingState.TURNING
        self.turn_steps_remaining = self.turn_steps
        self.turn_direction = -1.0 if degrees < 0 else 1.0  # Full steering
        self.turn_count += 1
        self.steering_history.clear()  # Clear smoothing buffer

        logger.info(
            f"Starting {'left' if degrees < 0 else 'right'} turn "
            f"({abs(degrees)} degrees, {self.turn_steps_remaining} steps)"
        )

    def _continue_turn(self) -> tuple:
        """
        Continue executing turn maneuver.

        Returns:
            Tuple of (steering, throttle)
        """
        self.turn_steps_remaining -= 1

        if self.turn_steps_remaining <= 0:
            # Turn complete, return to normal
            self.state = DrivingState.NORMAL
            self.steering_history.clear()
            logger.info("Turn complete, resuming normal driving")
            return 0.0, self.constant_throttle

        return self.turn_direction, self.turn_throttle

    @torch.no_grad()
    def predict_token(self, image: Image.Image) -> str:
        """
        Predict steering token from image using VLM.

        Args:
            image: PIL Image from simulator (120x160 RGB)

        Returns:
            Token string (A-G)
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

        # Validate token
        if token not in ACTION_TOKENS:
            logger.warning(f"Invalid token '{token}', defaulting to 'D'")
            token = "D"

        return token

    def predict(self, image: Image.Image) -> float:
        """
        Predict steering from image using VLM.

        Legacy method for backward compatibility. Returns steering value only.

        Args:
            image: PIL Image from simulator (120x160 RGB)

        Returns:
            Steering value in range [-1, 1]
        """
        token = self.predict_token(image)
        action_info = ACTION_TOKENS.get(token, ACTION_TOKENS["D"])

        if action_info["type"] == "turn":
            # For legacy compatibility, return extreme steering for turn tokens
            return -1.0 if action_info["degrees"] < 0 else 1.0
        else:
            return action_info["value"]

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
                f"Total turns: {self.turn_count}, "
                f"Avg inference: {avg_time*1000:.1f}ms"
            )

    def on_close(self) -> None:
        """Called when handler is closed."""
        pass
