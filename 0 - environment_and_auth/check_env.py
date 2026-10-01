#!/usr/bin/env python3
"""Environment and Google Cloud Authentication Health Check for AlphaEvolve Tutorial."""

import importlib.util
import os
import sys

# Color codes for pretty terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_status(component: str, ok: bool, details: str = "", warning: bool = False):
    symbol = f"{GREEN}[✓]{RESET}" if ok else (f"{YELLOW}[⚠]{RESET}" if warning else f"{RED}[✗]{RESET}")
    print(f"  {symbol} {BOLD}{component:<30}{RESET} : {details}")


def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major == 3 and minor >= 10)
    details = f"Python {major}.{minor}.{sys.version_info.micro} (Requires >= 3.10)"
    print_status("Python Version", ok, details)
    return ok


def check_packages():
    required = [
        ("tensorflow", "TensorFlow (Deep Learning backend)"),
        ("numpy", "NumPy (Array computing)"),
        ("dotenv", "python-dotenv (Environment configuration)"),
        ("matplotlib", "Matplotlib (Visualization & plotting)"),
        ("google.auth", "Google Auth (GCP Identity & Credentials)"),
    ]
    all_ok = True
    print(f"\n{BOLD}Checking Core Python Dependencies:{RESET}")
    for pkg, desc in required:
        installed = importlib.util.find_spec(pkg) is not None
        if not installed and pkg == "dotenv":
            installed = importlib.util.find_spec("dotenv") is not None
        if installed:
            print_status(pkg, True, f"Installed ({desc})")
        else:
            print_status(pkg, False, f"Missing! Run 'pip install -r ../requirements.txt'")
            all_ok = False

    # Check alpha_evolve package
    ae_installed = importlib.util.find_spec("alpha_evolve") is not None
    if ae_installed:
        print_status("alpha_evolve", True, "Installed (Google AlphaEvolve SDK)")
    else:
        print_status("alpha_evolve", False, "Missing! Run 'pip install -r ../requirements.txt'", warning=True)

    return all_ok


def check_env_file():
    print(f"\n{BOLD}Checking Configuration (.env):{RESET}")
    # Search in current directory, parent directory, and root
    search_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".env")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    ]
    
    env_path = next((p for p in search_paths if os.path.exists(p)), None)
    if not env_path:
        print_status(".env file", False, "Not found! Run './run.sh' to generate one from template.", warning=True)
        return False

    print_status(".env file", True, f"Found at {env_path}")
    from dotenv import dotenv_values
    config = dotenv_values(env_path)

    project_id = config.get("PROJECT_ID", "").strip()
    ge_app_id = config.get("GE_APP_ID", "").strip()

    if not project_id or "your-" in project_id:
        print_status("PROJECT_ID", False, "Not set in .env (replace placeholder with your GCP Project ID)", warning=True)
    else:
        print_status("PROJECT_ID", True, f"Configured ({project_id})")

    if not ge_app_id or "your-" in ge_app_id:
        print_status("GE_APP_ID", False, "Not set in .env (replace placeholder with Gemini Enterprise Engine ID)", warning=True)
    else:
        print_status("GE_APP_ID", True, f"Configured ({ge_app_id})")

    return True


def check_gcp_adc():
    print(f"\n{BOLD}Checking Google Cloud Application Default Credentials (ADC):{RESET}")
    try:
        import google.auth
        credentials, project = google.auth.default()
        print_status("GCP ADC Auth", True, f"Authenticated! Default Quota Project: {project or 'Not explicitly set'}")
        return True
    except Exception as e:
        print_status("GCP ADC Auth", False, f"Not authenticated ({e}). Run 'gcloud auth application-default login'", warning=True)
        return False


def main():
    print("=" * 75)
    print(f"{BOLD}Step 0: AlphaEvolve Workshop Health Check{RESET}")
    print("=" * 75)

    py_ok = check_python_version()
    pkg_ok = check_packages()
    env_ok = check_env_file()
    adc_ok = check_gcp_adc()

    print("\n" + "=" * 75)
    if py_ok and pkg_ok and adc_ok:
        print(f"{GREEN}{BOLD}✨ Environment is healthy and ready for the workshop!{RESET}")
        print(f"Next Step: {BOLD}cd '../1 - the_baseline_model' && ./run.sh{RESET}")
    else:
        print(f"{YELLOW}{BOLD}ℹ Note on workshop execution:{RESET}")
        print("  - Steps 1, 2, and 3 are completely local and run without GCP access.")
        print("  - Step 4 (Live Cloud Evolution) requires active GCP ADC & enabled APIs.")
    print("=" * 75)


if __name__ == "__main__":
    main()
