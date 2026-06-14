"""
setup_ollama.py

Purpose
-------
Automatically prepare Ollama environment for local development.

This script performs:

1. Check whether Ollama is installed
2. Install Ollama if possible
3. Start Ollama server
4. Verify Ollama health
5. Download embedding model
6. Download LLM model
7. Verify required models exist

Usage
-----
python scripts/setup_ollama.py

Required .env
-------------
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2

Supported Platforms
-------------------
Windows:
    Uses winget

Mac:
    Uses Homebrew

Linux:
    Uses official install script
"""

import os
import platform
import shutil
import subprocess
import time

import requests
from dotenv import load_dotenv

# =====================================================
# Configuration
# =====================================================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")


def run_command(command: list[str], check: bool = True):
    print(f"Running: {' '.join(command)}")
    return subprocess.run(command, check=check)

def is_ollama_installed() -> bool:
    """
    Check whether the 'ollama' command
    is available in system PATH.
    """
    return shutil.which("ollama") is not None

# =====================================================
# Installation Helpers
# =====================================================
def install_ollama_windows() -> bool:
    if shutil.which("winget") is None:
        print("Winget not found.")
        print("Please install Ollama manually:")
        print("https://ollama.com/download/windows")
        return False

    try:
        print("Installing Ollama using winget...")
        run_command(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
        )
        print("Ollama installed successfully.")
        return True

    except subprocess.CalledProcessError as error:
        print(f"Failed to install Ollama: {error}")
        print("Please install manually:")
        print("https://ollama.com/download/windows")
        return False


def install_ollama_mac() -> bool:
    if shutil.which("brew"):
        run_command(["brew", "install", "ollama"])
        return True

    print("Homebrew not found.")
    print("Please install Ollama manually:")
    print("https://ollama.com/download")
    return False


def install_ollama_linux() -> bool:
    run_command(
        [
            "bash",
            "-c",
            "curl -fsSL https://ollama.com/install.sh | sh",
        ]
    )
    return True


def install_ollama() -> bool:
    system = platform.system().lower()

    if "windows" in system:
        return install_ollama_windows()

    if "darwin" in system:
        return install_ollama_mac()

    if "linux" in system:
        return install_ollama_linux()

    print(f"Unsupported OS: {system}")
    return False


def wait_for_ollama_install() -> bool:
    for _ in range(30):
        if shutil.which("ollama"):
            return True
        time.sleep(2)

    return False

# =====================================================
# Ollama Health Checks
# =====================================================
def is_ollama_running() -> bool:
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5,
        )
        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def start_ollama():
    """
    Start Ollama local server.

    Default endpoint:

        http://localhost:11434

    Skip if already running.
    """
    if is_ollama_running():
        print("Ollama server is already running.")
        return

    print("Starting Ollama server...")

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    except FileNotFoundError:
        print("Cannot start Ollama because command was not found.")
        print("Please restart your terminal and run this script again.")
        raise

    for _ in range(30):
        if is_ollama_running():
            print("Ollama server is running.")
            return

        time.sleep(1)

    raise RuntimeError("Ollama server did not start.")

# =====================================================
# Model Management
# =====================================================
def get_installed_models() -> list[str]:
    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    return [
        model["name"]
        for model in data.get("models", [])
    ]


def is_model_installed(
    model_name: str,
    installed_models: list[str],
) -> bool:
    return any(
        item == model_name or item.startswith(f"{model_name}:")
        for item in installed_models
    )


def pull_model(model_name: str):
    """
    Download model from Ollama registry
    if not already installed locally.
    """
    installed_models = get_installed_models()

    if is_model_installed(model_name, installed_models):
        print(f"Model already exists: {model_name}")
        return

    print(f"Pulling model: {model_name}")
    run_command(["ollama", "pull", model_name])
    print(f"Model ready: {model_name}")

# =====================================================
# Status Reporting
# =====================================================
def print_ollama_status():
    print("\nOllama status")
    print("--------------------")

    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Installed: {is_ollama_installed()}")
    print(f"Server running: {is_ollama_running()}")

    if is_ollama_running():
        models = get_installed_models()

        print("\nInstalled models:")
        if models:
            for model in models:
                print(f"- {model}")
        else:
            print("- No models installed")


def verify_required_models():
    """
    Ensure both embedding model and
    LLM model defined in .env exist.
    """
    models = get_installed_models()

    missing_models = []

    if not is_model_installed(OLLAMA_EMBED_MODEL, models):
        missing_models.append(OLLAMA_EMBED_MODEL)

    if not is_model_installed(OLLAMA_LLM_MODEL, models):
        missing_models.append(OLLAMA_LLM_MODEL)

    if missing_models:
        raise RuntimeError(
            "Missing required Ollama models: "
            + ", ".join(missing_models)
        )

    print("\nRequired models are ready.")
    print(f"Embedding model: {OLLAMA_EMBED_MODEL}")
    print(f"LLM model: {OLLAMA_LLM_MODEL}")

# =====================================================
# Main Entry Point
# =====================================================
def main():
    """
    Setup flow

        Install Ollama
            ↓
        Start Server
            ↓
        Check Health
            ↓
        Pull Models
            ↓
        Verify Models
            ↓
        Ready
    """
    print("Setting up Ollama...")

    if not is_ollama_installed():
        print("Ollama is not installed.")

        installed = install_ollama()

        if not installed:
            return

        print("Waiting for Ollama installation...")

        if not wait_for_ollama_install():
            print(
                "Installation completed, but terminal PATH has not refreshed."
            )
            print(
                "Please restart PowerShell/Terminal and run this script again."
            )
            return

    print("Ollama is installed.")

    start_ollama()

    print_ollama_status()

    pull_model(OLLAMA_EMBED_MODEL)
    pull_model(OLLAMA_LLM_MODEL)

    verify_required_models()

    print_ollama_status()

    print("\nOllama setup completed successfully.")


if __name__ == "__main__":
    main()