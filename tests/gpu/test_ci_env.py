import subprocess


def test_ci_env_info():
    """CI environment verification for diagnostics."""
    for cmd in ["date", "hostname", "whoami", "id"]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{cmd}: {result.stdout.strip()}")
