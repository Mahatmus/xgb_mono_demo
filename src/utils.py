"""Utility functions for XGBoost progressive training."""


def parse_monotonic_constraints(mono_args: list[str]) -> dict[str, int]:
    """
    Parse monotonic constraint arguments.

    Args:
        mono_args: List of "feature:direction" strings, e.g., ["weight:-1", "age:+1"]
                   Direction can be +1, 1, -1. Defaults to +1 if omitted.

    Returns:
        Dict mapping feature name to constraint direction (+1 or -1)
    """
    constraints = {}
    for arg in mono_args:
        if ":" in arg:
            feat, direction = arg.rsplit(":", 1)
            direction = int(direction)
        else:
            feat = arg
            direction = 1
        if direction not in (-1, 1):
            raise ValueError(
                f"Invalid monotonic direction for {feat}: {direction}. Must be +1 or -1."
            )
        constraints[feat] = direction
    return constraints
