import yaml
import dotenv
from pathlib import Path
from typing import Dict, Any, Union

def load_config(config_path: Union[str, Path] = "config/agent.yaml") -> Dict[str, Any]:
    dotenv.load_dotenv()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    validate_config(config)
    return config

def validate_config(config: Dict[str, Any]) -> None:
    if not config:
        raise ValueError("Configuration cannot be empty.")
    if 'safety' not in config:
        raise ValueError("Missing 'safety' section in config.")
    safety = config['safety']
    required_safety_keys = ['max_motor_runtime_sec', 'motor_cooldown_sec', 'emergency_stop_timeout_sec']
    for key in required_safety_keys:
        if key not in safety:
            raise ValueError(f"Missing required safety parameter: {key}")
