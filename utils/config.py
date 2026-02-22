"""Centralised configuration for Sentinel360."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SUMO_HOME = os.environ.get('SUMO_HOME', '/opt/homebrew/opt/sumo/share/sumo')

# Simulation files (inside simulation/)
SIM_DIR = BASE_DIR / 'simulation'
NETWORK_FILE = str(SIM_DIR / 'accra.net.xml')
ROUTE_FILE = str(SIM_DIR / 'vehicles.rou.xml')
CONFIG_FILE = str(SIM_DIR / 'simulation.sumocfg')

# Model files (inside models/)
MODEL_DIR = str(BASE_DIR / 'models')
GHANA_GB_FILE = str(BASE_DIR / 'models' / 'ghana_gb_model.pkl')
PORTO_IF_FILE = str(BASE_DIR / 'models' / 'porto_if_model.pkl')
GHANA_SCALER_FILE = str(BASE_DIR / 'models' / 'ghana_scaler.pkl')
PORTO_SCALER_FILE = str(BASE_DIR / 'models' / 'porto_scaler.pkl')
FEATURES_FILE = str(BASE_DIR / 'models' / 'feature_names.pkl')
FUSION_CONFIG_FILE = str(BASE_DIR / 'models' / 'fusion_config.pkl')

STEP_LENGTH = 1.0
MAX_STEPS = 1000
UPDATE_INTERVAL = 5

COLORS = {'SAFE': (0, 255, 0), 'MEDIUM': (255, 255, 0), 'HIGH': (255, 0, 0)}
RISK_THRESHOLDS = {'SAFE': 0.3, 'MEDIUM': 0.7}
