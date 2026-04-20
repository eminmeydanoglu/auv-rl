"""Task-specific AUVRL extensions.

Tasks live in their own subpackages so each behavior can own its command,
observation, reward, and config layers without bleeding into the others.

Current task packages:

- ``auvrl.tasks.velocity``: working 6-DoF body-velocity tracking task
- ``auvrl.tasks.roll``: scaffold for the future roll-specialist task
"""

from . import roll, velocity

__all__ = ["roll", "velocity"]
