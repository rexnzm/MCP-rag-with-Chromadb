"""
Ollama Connection Diagnostic Tool
Run this to check if Ollama is properly configured and accessible.
"""

import requests
import sys
from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


def check_ollama_running():
    """Check if Ollama service is running."""
    print("=" * 60)
    print("1. Checking if Ollama service is running...")
    print("=" * 60)

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama service is running")
            return True
        else:
            print(f"✗ Ollama responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama service")
        print(f"  Make sure Ollama is running at {OLLAMA_BASE_URL}")
        print("\n  To start Ollama:")
        print("  - On Windows: Run Ollama from Start Menu or system tray")
        print("  - On Mac/Linux: Run 'ollama serve' in terminal")
        return False
    except requests.exceptions.Timeout:
        print("✗ Connection to Ollama timed out")
        return False
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        return False


def check_model_installed():
    """Check if the embedding model is installed."""
    print("\n" + "=" * 60)
    print("2. Checking if embedding model is installed...")
    print("=" * 60)

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            # List all installed models
            print(f"\nInstalled models ({len(models)}):")
            for model in models:
                model_name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_mb = size / (1024 * 1024)
                print(f"  - {model_name} ({size_mb:.1f} MB)")

            # Check for our specific model
            model_names = [m.get("name", "") for m in models]

            # Check various possible names
            possible_names = [
                EMBED_MODEL,
                "nomic-embed-text",
                "nomic-embed-text:latest"
            ]

            found = False
            for name in possible_names:
                if name in model_names:
                    print(f"\n✓ Model '{name}' is installed")
                    found = True
                    break

            if not found:
                print(f"\n✗ Model '{EMBED_MODEL}' is NOT installed")
                print("\n  To install the model, run:")
                print(f"  ollama pull nomic-embed-text")
                return False

            return True
        else:
            print(f"✗ Failed to get model list (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False


def test_embedding():
    """Test if embeddings can be generated."""
    print("\n" + "=" * 60)
    print("3. Testing embedding generation...")
    print("=" * 60)

    try:
        # Test with a simple string
        test_text = "This is a test sentence."

        payload = {
            "model": EMBED_MODEL.replace(":latest", ""),  # Try without :latest tag
            "prompt": test_text
        }

        print(f"Requesting embedding for: '{test_text}'")
        print(f"Using model: {payload['model']}")

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data.get("embedding", [])
            if embedding:
                print(f"✓ Successfully generated embedding")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  Sample values: {embedding[:5]}...")
                return True
            else:
                print("✗ Received empty embedding")
                return False
        else:
            print(f"✗ Embedding request failed (status: {response.status_code})")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("✗ Embedding request timed out")
        print("  This might indicate:")
        print("  - Ollama is overloaded or out of memory")
        print("  - The model is too large for your system")
        print("  - Try restarting Ollama")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Connection error during embedding: {e}")
        print("  The connection was forcibly closed")
        print("  This might indicate:")
        print("  - Ollama crashed or ran out of memory")
        print("  - System resources (RAM/CPU) exhausted")
        print("  - Firewall or antivirus blocking the connection")
        return False
    except Exception as e:
        print(f"✗ Error generating embedding: {e}")
        return False


def check_system_resources():
    """Check system resources."""
    print("\n" + "=" * 60)
    print("4. Checking system resources...")
    print("=" * 60)

    try:
        import psutil

        # Memory
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.percent}% used ({mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB)")

        if mem.percent > 90:
            print("  ⚠ Warning: Low memory available")
            print("  Ollama may struggle with large models")

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU: {cpu_percent}% used")

        if cpu_percent > 80:
            print("  ⚠ Warning: High CPU usage")

        # Disk
        disk = psutil.disk_usage(str(Path.home()))
        print(f"Disk: {disk.percent}% used ({disk.free / (1024**3):.1f} GB free)")

        if disk.percent > 95:
            print("  ⚠ Warning: Low disk space")

    except ImportError:
        print("  (Install 'psutil' for system resource monitoring)")
        print("  pip install psutil")
    except Exception as e:
        print(f"  Could not check system resources: {e}")


def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 60)
    print("OLLAMA CONNECTION DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Model: {EMBED_MODEL}")
    print("=" * 60)

    results = {
        "service": check_ollama_running(),
        "model": False,
        "embedding": False
    }

    if results["service"]:
        results["model"] = check_model_installed()

        if results["model"]:
            results["embedding"] = test_embedding()

    check_system_resources()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check.upper()}: {status}")

    if all_passed:
        print("\n✓ All checks passed! Ollama is properly configured.")
        print("  Your RAG system should work correctly.")
    else:
        print("\n✗ Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("1. Start/Restart Ollama service")
        print("2. Pull the model: ollama pull nomic-embed-text")
        print("3. Check available RAM (Ollama needs ~4GB for embedding models)")
        print("4. Restart your computer if Ollama is unresponsive")
        print("5. Try a smaller model if you have limited RAM")

    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
