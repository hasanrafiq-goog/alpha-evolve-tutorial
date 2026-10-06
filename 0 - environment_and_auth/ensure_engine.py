#!/usr/bin/env python3
"""Auto-verifies, resolves, or creates the Gemini Enterprise Engine in Google Cloud."""

import os
import sys
import google.auth
from google.cloud import discoveryengine_v1beta as discoveryengine
from dotenv import dotenv_values

# Styling
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def update_env_file(env_path: str, key: str, value: str):
    """Updates a single key in the .env file."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        lines = f.readlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(new_lines)


def main():
    search_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".env")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
    ]
    env_path = next((p for p in search_paths if os.path.exists(p)), None)
    if not env_path:
        print(f"{YELLOW}Warning: .env file not found.{RESET}")
        return

    config = dotenv_values(env_path)
    project_id = config.get("PROJECT_ID", "").strip()
    target_engine_id = config.get("GE_APP_ID", "").strip()

    if not project_id or "your-" in project_id:
        print(f"{YELLOW}PROJECT_ID not configured in .env. Skipping engine verification.{RESET}")
        return

    print(f"{CYAN}Checking Gemini Enterprise Engine in project {BOLD}{project_id}{RESET}...")

    credentials, _ = google.auth.default()
    client = discoveryengine.EngineServiceClient(credentials=credentials)
    parent = f"projects/{project_id}/locations/global/collections/default_collection"

    try:
        request = discoveryengine.ListEnginesRequest(parent=parent)
        engines = list(client.list_engines(request=request))
    except Exception as e:
        print(f"{YELLOW}Note: Unable to list Discovery Engine apps automatically ({e}).{RESET}")
        sys.exit(1)

    # Check for exact match
    resolved_id = None
    for e in engines:
        eid = e.name.split("/")[-1]
        if eid == target_engine_id:
            resolved_id = eid
            break

    # Check for prefix or display_name match (e.g. Google Cloud appends timestamp _1767869485122)
    if not resolved_id:
        for e in engines:
            eid = e.name.split("/")[-1]
            if eid.startswith(target_engine_id) or e.display_name == target_engine_id:
                resolved_id = eid
                print(f"{GREEN}[✓] Discovered full Engine Resource ID: {BOLD}{resolved_id}{RESET} (Display: {e.display_name})")
                print(f"    Updating .env with the full Engine ID...")
                update_env_file(env_path, "GE_APP_ID", resolved_id)
                # Also update current local .env if present
                local_env = os.path.join(os.path.dirname(__file__), ".env")
                if os.path.exists(local_env):
                    update_env_file(local_env, "GE_APP_ID", resolved_id)
                break

    # No match but apps exist: use the only one if GE_APP_ID is unset, otherwise ask which one to use
    if not resolved_id and engines:
        ids = [e.name.split("/")[-1] for e in engines]
        if len(ids) > 1 or "your-" not in target_engine_id:
            print(f"{YELLOW}GE_APP_ID={target_engine_id} not found. Existing apps: {', '.join(ids)}. Set GE_APP_ID in .env to one of them.{RESET}")
            sys.exit(1)
        resolved_id = ids[0]
        update_env_file(env_path, "GE_APP_ID", resolved_id)

    # If no match found, create a new engine automatically!
    if not resolved_id and not engines:
        print(f"{YELLOW}No Gemini Enterprise Engine found. Auto-creating a new engine...{RESET}")
        new_engine_id = target_engine_id if target_engine_id and "your-" not in target_engine_id else "alphaevolve-engine"
        try:
            new_engine = discoveryengine.Engine(
                display_name=new_engine_id,
                solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
                app_type=discoveryengine.Engine.AppType.APP_TYPE_INTRANET,
                search_engine_config=discoveryengine.Engine.SearchEngineConfig(
                    search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
                    search_add_ons=[discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM],
                ),
            )
            create_req = discoveryengine.CreateEngineRequest(
                parent=parent,
                engine=new_engine,
                engine_id=new_engine_id,
            )
            operation = client.create_engine(request=create_req)
            print(f"Creating engine {new_engine_id}... (Operation: {operation.operation.name})")
            created = operation.result(timeout=180)
            resolved_id = created.name.split("/")[-1]
            print(f"{GREEN}[✓] Engine created successfully: {BOLD}{resolved_id}{RESET}")
            update_env_file(env_path, "GE_APP_ID", resolved_id)
        except Exception as create_err:
            print(f"{YELLOW}Could not auto-create engine: {create_err}{RESET}")
            sys.exit(1)

    if resolved_id:
        print(f"{GREEN}[✓] Gemini Enterprise Engine verified: {BOLD}{resolved_id}{RESET}")


if __name__ == "__main__":
    main()
