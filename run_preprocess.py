import os
import sys
import argparse
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess_agent import run_route_discovery
# Import the new enrichment function
from src.enrichment_agent import run_enrichment

def main():
    parser = argparse.ArgumentParser(description="Route Discovery & Enrichment Pipeline")
    parser.add_argument("--log", action="store_true", help="Enable detailed logging")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip discovery, only run enrichment on existing routes.json")
    args = parser.parse_args()
    
    # PHASE 1: DISCOVERY
    if not args.skip_discovery:
        print("\n=== PHASE 1: ROUTE DISCOVERY AGENT ===")
        run_route_discovery(enable_logging=args.log)
    else:
        print("\n=== PHASE 1: SKIPPED ===")

    # PHASE 2: ENRICHMENT
    if os.path.exists("routes.json"):
        print("\n=== PHASE 2: USE CASE ENRICHMENT AGENT ===")
        try:
            asyncio.run(run_enrichment("routes.json"))
        except Exception as e:
            print(f"❌ Enrichment Phase Failed: {e}")
    else:
        print("⚠️ Warning: routes.json not found. Cannot proceed to enrichment.")

if __name__ == "__main__":
    main()