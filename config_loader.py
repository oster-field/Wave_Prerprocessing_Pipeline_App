"""
Configuration loader for Wave Preprocessing Pipeline
Loads and validates configuration from config.json
"""

import json
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG = {
    "application": {
        "name": "Wave Preprocessing Pipeline",
        "version": "1.0.0",
        "window_title": "🌊 Sakhalin Wave Processor"
    },
    "data_processing": {
        "max_file_size_mb": 500,
        "visualization_target_points": 5000,
        "spectrum_target_points": 100000,
        "default_sensor_frequency": 8,
        "reading_duration_seconds": 1200
    },
    "file_formats": {
        "info_extensions": [".dat", ".txt"],
        "data_extensions": [".dat", ".txt", ".npy"],
        "output_format": "csv"
    },
    "output": {
        "directory": "Output",
        "step1_filename": "Step1_TXTtoCSV.csv",
        "visualization_cache": "Step1_Visualization.csv"
    },
    "performance": {
        "chunk_size": 100000,
        "parallel_processing": False,
        "memory_limit_mb": 2048
    },
    "ui": {
        "theme": "default",
        "show_progress_dialog": True,
        "enable_tooltips": True
    }
}


class Config:
    """Configuration manager for the application"""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager

        Args:
            config_file: Path to config.json file (default: config.json in script directory)
        """
        self._config = DEFAULT_CONFIG.copy()

        if config_file is None:
            # Look for config.json in script directory
            script_dir = Path(__file__).parent
            config_file = script_dir / "config.json"

        # Load from file if exists
        if Path(config_file).exists():
            try:
                self.load_from_file(config_file)
            except Exception as e:
                print(f"Warning: Could not load config from {config_file}: {e}")
                print("Using default configuration")

    def load_from_file(self, filepath: str):
        """
        Load configuration from JSON file

        Args:
            filepath: Path to configuration file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            user_config = json.load(f)

        # Merge with defaults (user config overrides defaults)
        self._merge_config(user_config)

    def _merge_config(self, user_config: Dict[str, Any]):
        """Recursively merge user config with defaults"""
        for key, value in user_config.items():
            if key in self._config:
                if isinstance(value, dict) and isinstance(self._config[key], dict):
                    self._merge_dict(self._config[key], value)
                else:
                    self._config[key] = value
            else:
                self._config[key] = value

    def _merge_dict(self, default: Dict, user: Dict):
        """Merge two dictionaries"""
        for key, value in user.items():
            if key in default and isinstance(value, dict) and isinstance(default[key], dict):
                self._merge_dict(default[key], value)
            else:
                default[key] = value

    def get(self, *keys, default=None):
        """
        Get configuration value by nested keys

        Args:
            *keys: Nested keys to traverse (e.g., 'data_processing', 'max_file_size_mb')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value):
        """
        Set configuration value by nested keys

        Args:
            *keys: Nested keys to traverse
            value: Value to set
        """
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def save_to_file(self, filepath: str = None):
        """
        Save configuration to JSON file

        Args:
            filepath: Path to save configuration (default: config.json in script directory)
        """
        if filepath is None:
            script_dir = Path(__file__).parent
            filepath = script_dir / "config.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    @property
    def max_file_size_mb(self):
        """Get maximum file size in MB"""
        return self.get('data_processing', 'max_file_size_mb', default=500)

    @property
    def visualization_target_points(self):
        """Get target points for visualization"""
        return self.get('data_processing', 'visualization_target_points', default=5000)

    @property
    def spectrum_target_points(self):
        """Get target points for spectrum visualization"""
        return self.get('data_processing', 'spectrum_target_points', default=100000)

    @property
    def output_directory(self):
        """Get output directory name"""
        return self.get('output', 'directory', default='Output')

    @property
    def chunk_size(self):
        """Get chunk size for data processing"""
        return self.get('performance', 'chunk_size', default=100000)


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config(config_file: str = None):
    """Reload configuration from file"""
    global _config
    _config = Config(config_file)
    return _config
