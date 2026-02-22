"""HybridMonitor – real-time risk fusion (0.7xGB + 0.3xIF)."""
import numpy as np
from collections import deque, defaultdict

from core.ml_inference import load_model, predict_ghana_gb, predict_porto_if
from core.expert_rules import rule_based_risk_score
from utils.gps_utils import haversine
from utils.config import (
    GHANA_GB_FILE, PORTO_IF_FILE,
    GHANA_SCALER_FILE, PORTO_SCALER_FILE,
    FEATURES_FILE, FUSION_CONFIG_FILE,
    COLORS,
)


class HybridMonitor:
    """Hybrid model: weight_gb x Ghana GB + weight_if x Porto IF fusion."""

    def __init__(self):
        self.ghana_gb = load_model(GHANA_GB_FILE)
        if self.ghana_gb:
            print(f"\u2705 Ghana GB loaded: {type(self.ghana_gb).__name__}")

        self.porto_if = load_model(PORTO_IF_FILE)
        if self.porto_if:
            print(f"\u2705 Porto IF loaded: {type(self.porto_if).__name__}")

        self.ghana_scaler = load_model(GHANA_SCALER_FILE)
        self.porto_scaler = load_model(PORTO_SCALER_FILE)
        if self.ghana_scaler and self.porto_scaler:
            print(f"\u2705 Scalers loaded (Ghana: {type(self.ghana_scaler).__name__}, "
                  f"Porto: {type(self.porto_scaler).__name__})")

        self.features = load_model(FEATURES_FILE)
        self.fusion_config = load_model(FUSION_CONFIG_FILE) or {}
        if self.features:
            print(f"\u2705 Features loaded ({len(self.features)} features)")

        self.weight_gb = self.fusion_config.get('weight_gb', 0.7)
        self.weight_if = self.fusion_config.get('weight_if', 0.3)
        self.use_rule_based = False

        if self.ghana_gb and self.porto_if:
            self.mode = 'hybrid'
            print(f"\u2705 Hybrid Monitor ready")
            print(f"   Mode: HYBRID FUSION")
        elif self.porto_if:
            self.mode = 'porto_only'
            self.weight_if, self.weight_gb = 1.0, 0.0
            print(f"\u2705 Monitor ready (Porto IF only)")
        elif self.ghana_gb:
            self.mode = 'ghana_only'
            self.weight_gb, self.weight_if = 1.0, 0.0
            print(f"\u2705 Monitor ready (Ghana GB only)")
        else:
            self.mode = 'rule_based'
            self.use_rule_based = True
            print(f"\u26a0\ufe0f  No ML models loaded, using rule-based")

        print(f"   Fusion: {self.weight_gb}\u00d7GB + {self.weight_if}\u00d7IF")

        self.vehicles = defaultdict(lambda: {
            'speeds': deque(maxlen=10), 'positions': deque(maxlen=10),
            'accels': deque(maxlen=10), 'distance': 0.0, 'stops': 0,
            'last_speed': 0, 'trip_start': None,
            'risk_score': 0.0, 'risk_level': 'SAFE',
        })

    # ── telemetry ────────────────────────────────────────────────────────

    def update_vehicle(self, veh_id, speed, position, timestamp):
        data = self.vehicles[veh_id]
        if data['trip_start'] is None:
            data['trip_start'] = timestamp
        speed_kmh = speed * 3.6
        data['speeds'].append(speed_kmh)
        data['positions'].append(position)
        if len(data['speeds']) >= 2:
            data['accels'].append(data['speeds'][-1] - data['speeds'][-2])
        if len(data['positions']) >= 2:
            data['distance'] += haversine(data['positions'][-2], data['positions'][-1])
        if speed_kmh < 1.0 and data['last_speed'] > 2.0:
            data['stops'] += 1
        data['last_speed'] = speed_kmh

    # ── feature engineering ──────────────────────────────────────────────

    def _build_features(self, data, trip_duration):
        return {
            'speed': data['speeds'][-1],
            'acceleration': data['accels'][-1] if data['accels'] else 0,
            'acceleration_variation': float(np.std(data['accels'])) if len(data['accels']) > 1 else 0,
            'trip_duration': trip_duration,
            'trip_distance': data['distance'],
            'stop_events': data['stops'],
            'road_encoded': 0,
            'weather_encoded': 0,
            'traffic_encoded': 1,
            'hour': 10,
            'month': 2,
            'avg_speed': data['distance'] / (trip_duration / 3600 + 0.001),
            'stops_per_km': data['stops'] / (data['distance'] + 0.1),
            'accel_abs': abs(data['accels'][-1]) if data['accels'] else 0,
            'speed_normalized': data['speeds'][-1] / 100,
            'speed_squared': data['speeds'][-1] ** 2,
            'is_rush_hour': 0,
            'is_night': 0,
        }

    # ── classification helpers ───────────────────────────────────────────

    def _classify(self, score):
        safe = self.fusion_config.get('threshold_safe', 0.3)
        high = self.fusion_config.get('threshold_high', 0.7)
        if score < safe:
            return 'SAFE', COLORS['SAFE']
        elif score < high:
            return 'MEDIUM', COLORS['MEDIUM']
        return 'HIGH', COLORS['HIGH']

    def _rule_based_risk(self, data, trip_duration):
        score = rule_based_risk_score(
            data['speeds'][-1], list(data['accels']),
            data['stops'], trip_duration,
        )
        level, color = self._classify(score)
        data['risk_score'] = score
        data['risk_level'] = level
        return score, level, color

    # ── main prediction ──────────────────────────────────────────────────

    def predict_risk(self, veh_id, timestamp):
        data = self.vehicles[veh_id]
        if len(data['speeds']) < 5:
            return 0.0, 'SAFE', COLORS['SAFE']

        trip_duration = timestamp - data['trip_start']

        if self.use_rule_based:
            return self._rule_based_risk(data, trip_duration)

        try:
            features = self._build_features(data, trip_duration)
            feature_array = [features[f] for f in self.features]
            risk_score = 0.0

            if self.ghana_gb and self.ghana_scaler and self.weight_gb > 0:
                risk_score += self.weight_gb * predict_ghana_gb(
                    feature_array, self.ghana_gb, self.ghana_scaler)

            if self.porto_if and self.porto_scaler and self.weight_if > 0:
                risk_score += self.weight_if * predict_porto_if(
                    feature_array, self.porto_if, self.porto_scaler)
        except Exception:
            return self._rule_based_risk(data, trip_duration)

        level, color = self._classify(risk_score)
        data['risk_score'] = risk_score
        data['risk_level'] = level
        return risk_score, level, color

    # ── statistics ───────────────────────────────────────────────────────

    def get_statistics(self):
        if not self.vehicles:
            return None
        counts = {'SAFE': 0, 'MEDIUM': 0, 'HIGH': 0}
        for d in self.vehicles.values():
            counts[d['risk_level']] += 1
        return {
            'total': len(self.vehicles),
            'safe': counts['SAFE'],
            'medium': counts['MEDIUM'],
            'high': counts['HIGH'],
            'avg_risk': float(np.mean([d['risk_score'] for d in self.vehicles.values()])),
        }
