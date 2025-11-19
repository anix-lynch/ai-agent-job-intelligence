#!/usr/bin/env python3
"""
Universal secret loader that works in both sandboxed and non-sandboxed environments.
Falls back from global config to project root .env file.
"""

import os
import sys
from pathlib import Path

def get_secret(key: str, default: str = None) -> str:
    """
    Get a secret value from multiple possible locations.
    
    Priority order:
    1. Environment variable (already set)
    2. ~/.config/secrets/global.env (if accessible)
    3. .env in project root (always accessible)
    4. Default value (if provided)
    5. Raise error if not found
    
    Args:
        key: The secret key name
        default: Optional default value if not found
        
    Returns:
        The secret value
        
    Raises:
        ValueError: If secret not found and no default provided
    """
    # 1. Check if already in environment
    if key in os.environ:
        return os.environ[key]
    
    # 2. Try global secrets file
    global_secrets = Path.home() / ".config" / "secrets" / "global.env"
    if global_secrets.exists():
        try:
            with open(global_secrets, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip('"\'')
        except (PermissionError, IOError):
            pass  # Fall through to next method
    
    # 3. Try project root .env
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip('"\'')
    
    # 4. Return default if provided
    if default is not None:
        return default
    
    # 5. Raise error if not found
    raise ValueError(f"Secret '{key}' not found in environment, global.env, or .env file")

def load_secrets_to_env():
    """Load all secrets from available sources into os.environ"""
    secrets_sources = [
        Path.home() / ".config" / "secrets" / "global.env",
        Path(__file__).parent.parent / ".env"
    ]
    
    for secrets_file in secrets_sources:
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip('"\'')
            except (PermissionError, IOError):
                continue

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) != 2:
        print("Usage: python get_secret.py <secret_key>")
        sys.exit(1)
    
    try:
        secret = get_secret(sys.argv[1])
        print(secret)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
