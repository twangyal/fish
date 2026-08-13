import sys
from unittest.mock import MagicMock

try:
    import yaml
except ImportError:
    mock_yaml = MagicMock()
    def fake_safe_load(f):
        content = f.read()
        if "safety:" in content:
            return {
                "safety": {
                    "max_motor_runtime_sec": 5.0,
                    "motor_cooldown_sec": 2.0,
                    "emergency_stop_timeout_sec": 1.0
                }
            }
        return {}
    mock_yaml.safe_load = fake_safe_load
    sys.modules['yaml'] = mock_yaml

import unittest
from pathlib import Path
import tempfile
import os
from config.loader import load_config, validate_config

class TestConfig(unittest.TestCase):
    def test_load_config_valid(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            tmp.write('''
safety:
  max_motor_runtime_sec: 5.0
  motor_cooldown_sec: 2.0
  emergency_stop_timeout_sec: 1.0
''')
            tmp_name = tmp.name
        
        try:
            config = load_config(Path(tmp_name))
            self.assertEqual(config['safety']['max_motor_runtime_sec'], 5.0)
        finally:
            os.remove(tmp_name)

    def test_validate_config_missing_safety(self):
        with self.assertRaisesRegex(ValueError, "Missing 'safety' section"):
            validate_config({"other": "value"})

    def test_validate_config_missing_key(self):
        with self.assertRaisesRegex(ValueError, "Missing required safety parameter"):
            validate_config({"safety": {"max_motor_runtime_sec": 5.0}})
