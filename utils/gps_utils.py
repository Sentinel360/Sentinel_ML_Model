"""GPS / coordinate utilities."""
from math import radians, cos, sin, asin, sqrt


def haversine(pos1: tuple, pos2: tuple) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(radians, [*pos1, *pos2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def cumulative_distance(positions: list) -> float:
    """Total distance in km from a list of (lon, lat) positions."""
    total = 0.0
    for i in range(1, len(positions)):
        total += haversine(positions[i - 1], positions[i])
    return total
