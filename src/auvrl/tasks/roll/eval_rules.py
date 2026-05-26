from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROLL_EVAL_RULES_FORMAT = "auvrl.roll_eval_rules.v1"


def _json_safe(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def capture_roll_eval_rules(env: Any, *, mode: str = "current") -> dict[str, Any]:
    rewards: dict[str, Any] = {}
    for name in getattr(env.reward_manager, "active_terms", []):
        cfg = env.reward_manager.get_term_cfg(name)
        rewards[name] = {
            "weight": _json_safe(getattr(cfg, "weight", None)),
            "params": _json_safe(getattr(cfg, "params", {})),
        }
    terminations: dict[str, Any] = {}
    for name in getattr(env.termination_manager, "active_terms", []):
        cfg = env.termination_manager.get_term_cfg(name)
        terminations[name] = {
            "params": _json_safe(getattr(cfg, "params", {})),
        }
    return {
        "format": ROLL_EVAL_RULES_FORMAT,
        "mode": mode,
        "episode_length_s": _json_safe(getattr(env.cfg, "episode_length_s", None)),
        "step_dt_s": _json_safe(getattr(env, "step_dt", None)),
        "rewards": rewards,
        "terminations": terminations,
    }


def load_roll_eval_rules(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        rules = json.load(file)
    if not isinstance(rules, dict):
        raise ValueError(f"Expected roll eval rules mapping in {path}.")
    return rules


def write_roll_eval_rules(path: Path, rules: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(rules), file, indent=2, sort_keys=True)
        file.write("\n")


def apply_roll_eval_rules_to_cfg(cfg: Any, rules: dict[str, Any] | None) -> None:
    if not rules:
        return
    episode_length_s = rules.get("episode_length_s")
    if episode_length_s is not None:
        cfg.episode_length_s = float(episode_length_s)
    for name, item in rules.get("rewards", {}).items():
        if name not in cfg.rewards:
            continue
        term_cfg = cfg.rewards[name]
        if isinstance(item, dict) and item.get("weight") is not None:
            term_cfg.weight = float(item["weight"])
        params = item.get("params", {}) if isinstance(item, dict) else {}
        if isinstance(params, dict):
            term_cfg.params.update(params)
    for name, item in rules.get("terminations", {}).items():
        if name not in cfg.terminations:
            continue
        params = item.get("params", {}) if isinstance(item, dict) else {}
        if isinstance(params, dict):
            cfg.terminations[name].params.update(params)
