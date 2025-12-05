import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess_agent import run_route_discovery

def main():
    parser = argparse.ArgumentParser(description="Route Discovery Preprocessing Agent")
    parser.add_argument("--log", action="store_true", help="Enable detailed logging to file")
    args = parser.parse_args()
    
    print("--- STARTING PRE-PROCESS: ROUTE MAPPING ---")
    
    run_route_discovery(enable_logging=args.log)
    
    if os.path.exists("routes.json"):
        import json
        with open("routes.json", "r") as f:
            data = json.load(f)
        print(f"📊 Summary: Found {len(data)} routes.")
    else:
        print("⚠️ Warning: routes.json was not created. The agent might have failed to find routes.")

if __name__ == "__main__":
    main()