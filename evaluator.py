import os
import subprocess
import asyncio
import time
from typing import Set
from datasets import load_dataset
from unidiff import PatchSet
import pandas as pd
from langchain_core.messages import HumanMessage

# Imports
from src.local_tools import LocalTools
from src.agent import build_graph

# --- HELPER: VERBOSE SUBPROCESS ---
def run_command(args, cwd=None, description=""):
    """Runs a shell command with timing logs."""
    # print(f"   ⏳ {description}...", end="", flush=True) # Optional: reduced noise
    start = time.time()
    try:
        subprocess.run(
            args, 
            cwd=cwd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        elapsed = time.time() - start
        # print(f" Done ({elapsed:.1f}s)")
    except subprocess.CalledProcessError:
        print(f" ❌ Failed to run: {' '.join(args)}")
        raise

def get_gold_files(patch_text: str) -> Set[str]:
    patch = PatchSet(patch_text)
    return {p.path for p in patch}

async def run_retrieval_session(repo_path: str, issue_text: str) -> Set[str]:
    # print(f"   🤖 Initializing Agent in {os.path.basename(repo_path)}...")
    original_cwd = os.getcwd()
    os.chdir(repo_path) 
    
    read_files = set()
    
    try:
        tools = [
            LocalTools.read_file,
            LocalTools.list_directory,
            LocalTools.search_code,
            LocalTools.get_code_symbols
        ]
        
        graph = build_graph(tools)
        config = {
            "configurable": {"thread_id": "eval"},
            "recursion_limit": 15
        }
        
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=f"Locate the code responsible for: {issue_text}")]},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            
            # LOGGING: Tool Execution
            if kind == "on_tool_start":
                tool_name = event["name"]
                # Only log read_file to keep output clean
                if tool_name == "read_file":
                    args = event["data"].get("input", {})
                    path = args.get("path", "")
                    print(f"      👀 Reading: {path}")
                    read_files.add(path)
                elif tool_name == "search_code_lexical":
                    query = event["data"].get("input", {}).get("query", "")
                    print(f"      🔎 Searching: '{query}'")

    except Exception as e:
        if "recursion limit" in str(e).lower():
            print(f"      ⚠️  Hit Step Limit (15).")
        else:
            print(f"      ❌ Agent Error: {e}")
    finally:
        os.chdir(original_cwd)
        
    return read_files

async def evaluate_swe_lite(limit=5):
    print("📥 Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    results = []
    # Persistent cache directory
    work_dir = os.path.abspath("./benchmark_scratch")
    os.makedirs(work_dir, exist_ok=True)
    print(f"📂 Persistent Cache: {work_dir}")

    count = 0
    for row in dataset:
        if count >= limit: break
        
        instance_id = row['instance_id']
        repo_url = f"https://github.com/{row['repo']}"
        repo_name = row['repo'].split("/")[-1] # e.g., 'django'
        base_commit = row['base_commit']
        
        print(f"\n[{count+1}/{limit}] 🧪 {instance_id}")
        
        # 1. Prepare Repository (Smart Caching)
        instance_path = os.path.join(work_dir, repo_name)
        
        try:
            if not os.path.exists(instance_path):
                print(f"   ⬇️  Cloning {repo_name} (First time only)...")
                run_command(["git", "clone", repo_url, instance_path], description="Cloning")
            else:
                print(f"   ⚡ Using cached {repo_name}...")

            # 2. Reset and Checkout
            # Clean up any mess from previous runs
            run_command(["git", "reset", "--hard", "HEAD"], cwd=instance_path)
            run_command(["git", "clean", "-fdx"], cwd=instance_path)
            # Switch to the specific commit for this test
            run_command(["git", "checkout", base_commit], cwd=instance_path, description="Checkout")
            
            # 3. Parse Gold Solution
            gold_files = get_gold_files(row['patch'])

            # 4. Run Agent
            start_time = time.time()
            agent_files_raw = await run_retrieval_session(instance_path, row['problem_statement'])
            duration = time.time() - start_time
            
            # --- FIX: NORMALIZE PATHS ---
            # Remove leading "./" if the agent added it
            agent_files = {f.lstrip("./") for f in agent_files_raw}

            # 5. Results
            matches = gold_files.intersection(agent_files)
            success = len(matches) > 0
            
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} Result: {len(matches)}/{len(gold_files)} files found in {duration:.1f}s")
            
            if not success:
                 print(f"      Expected: {list(gold_files)}")
                 print(f"      Got:      {list(agent_files)}")

            results.append({
                "id": instance_id, 
                "success": success, 
                "matches": len(matches),
                "gold": list(gold_files)
            })
            count += 1
            
        except Exception as e:
            print(f"   ❌ Critical Setup Error: {e}")
            continue

    # Summary Table
    if results:
        df = pd.DataFrame(results)
        print("\n\n📊 ================= SUMMARY =================")
        print(df[["id", "success", "matches"]])
        print(f"🏆 Accuracy: {df['success'].mean() * 100:.2f}%")
        print("==========================================")

if __name__ == "__main__":
    asyncio.run(evaluate_swe_lite(limit=1))
