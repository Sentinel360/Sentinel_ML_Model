"""Rule-based risk scoring fallback (no ML models required)."""
import numpy as np


def rule_based_risk_score(current_speed, accels, stops, trip_duration):
    """Compute a 0-1 risk score using hand-crafted thresholds.

    Parameters
    ----------
    current_speed : float   – latest speed in km/h
    accels        : list    – recent acceleration deltas
    stops         : int     – number of full-stop events
    trip_duration : float   – seconds since trip start
    """
    risk = 0.0

    if current_speed > 80:
        risk += 0.3
    elif current_speed > 60:
        risk += 0.15

    if accels:
        recent = abs(accels[-1])
        if recent > 5.0:
            risk += 0.25
        elif recent > 3.0:
            risk += 0.15

        std = float(np.std(accels))
        if std > 4.0:
            risk += 0.2
        elif std > 2.5:
            risk += 0.1

    dur = max(1, trip_duration)
    stops_per_min = (stops / dur) * 60
    if stops_per_min > 2.0:
        risk += 0.15
    elif stops_per_min > 1.0:
        risk += 0.08

    return min(1.0, risk)
