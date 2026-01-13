"""Evaluation metrics for obstacle avoidance."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""

    episode_id: int
    total_steps: int = 0
    collision_count: int = 0
    turn_count: int = 0
    distance_traveled: float = 0.0
    avg_speed: float = 0.0
    cte_values: List[float] = field(default_factory=list)

    @property
    def cte_mean(self) -> float:
        """Mean cross-track error."""
        return float(np.mean(self.cte_values)) if self.cte_values else 0.0

    @property
    def cte_max(self) -> float:
        """Maximum absolute cross-track error."""
        return float(np.max(np.abs(self.cte_values))) if self.cte_values else 0.0

    @property
    def survival_rate(self) -> float:
        """Steps without collision / total steps."""
        return self.total_steps

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "total_steps": self.total_steps,
            "collision_count": self.collision_count,
            "turn_count": self.turn_count,
            "distance_traveled": self.distance_traveled,
            "avg_speed": self.avg_speed,
            "cte_mean": self.cte_mean,
            "cte_max": self.cte_max,
        }


@dataclass
class EvaluationSummary:
    """Summary metrics across all evaluation episodes."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add_episode(self, episode: EpisodeMetrics) -> None:
        """Add an episode to the summary."""
        self.episodes.append(episode)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.episodes)

    @property
    def avg_episode_length(self) -> float:
        """Average episode length in steps."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_steps for e in self.episodes]))

    @property
    def total_collisions(self) -> int:
        """Total collisions across all episodes."""
        return sum(e.collision_count for e in self.episodes)

    @property
    def collision_rate(self) -> float:
        """Collisions per 1000 steps."""
        total_steps = sum(e.total_steps for e in self.episodes)
        if total_steps == 0:
            return 0.0
        return (self.total_collisions / total_steps) * 1000

    @property
    def avg_distance(self) -> float:
        """Average distance traveled per episode."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.distance_traveled for e in self.episodes]))

    @property
    def total_turns(self) -> int:
        """Total turns across all episodes."""
        return sum(e.turn_count for e in self.episodes)

    @property
    def turn_rate(self) -> float:
        """Turns per 1000 steps."""
        total_steps = sum(e.total_steps for e in self.episodes)
        if total_steps == 0:
            return 0.0
        return (self.total_turns / total_steps) * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "num_episodes": self.num_episodes,
            "avg_episode_length": self.avg_episode_length,
            "total_collisions": self.total_collisions,
            "collision_rate_per_1k_steps": self.collision_rate,
            "avg_distance_traveled": self.avg_distance,
            "total_turns": self.total_turns,
            "turn_rate_per_1k_steps": self.turn_rate,
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Episodes: {self.num_episodes}")
        print(f"Avg Episode Length: {self.avg_episode_length:.1f} steps")
        print(f"Total Collisions: {self.total_collisions}")
        print(f"Collision Rate: {self.collision_rate:.2f} per 1k steps")
        print(f"Avg Distance: {self.avg_distance:.1f} units")
        print(f"Total Turns: {self.total_turns}")
        print(f"Turn Rate: {self.turn_rate:.2f} per 1k steps")
        print("=" * 50 + "\n")
