"""The user-requested task shapes, without freezing a training protocol."""
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "multiscale"
DEV_START = "1980-12-12"
DEV_END = "2026-06-30"
NY = "America/New_York"
TARGETS = ("future_close", "future_log_return")
MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")


@dataclass(frozen=True)
class Profile:
    name: str
    frequency: str
    horizons: tuple[int, ...]
    lookbacks: tuple[int, ...]
    primary_horizon: int
    minimum_history_years: float


PROFILES = (
    Profile("annual", "month", (12,), (12, 24, 36, 60), 12, 15.0),
    Profile("quarterly", "month", (3,), (6, 12, 24, 36), 3, 10.0),
    Profile("monthly", "day", (21,), (21, 63, 126, 252), 21, 5.0),
    Profile("hourly", "hour", (1, 4, 8), (12, 24, 48, 96), 8, 1.0),
    Profile("minute", "minute", (5, 15, 30, 60), (30, 60, 120, 240, 480), 60, 0.5),
)


def tasks():
    return [
        {"profile": profile.name, "frequency": profile.frequency,
         "horizon": horizon, "target": target,
         "lookbacks": list(profile.lookbacks),
         "primary": horizon == profile.primary_horizon}
        for profile in PROFILES for horizon in profile.horizons for target in TARGETS
    ]
