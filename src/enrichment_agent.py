import json
import os
import asyncio
import time
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# --- DATA MODELS ---
class RouteEnrichment(BaseModel):
    description: str = Field(description="A high-level, executive summary of the page's specific purpose.")
    usecases: List[str] = Field(description="A list of unique, intent-based functionalities specific to this route.")

# --- PROMPTS ---
ANALYSIS_SYSTEM_PROMPT = """
You are a Senior Product Manager and QA Architect.
Your goal is to analyze React Component code and extract its **Unique Value Proposition** and **Core Intent**.

### INSTRUCTIONS

1. **Focus on Intent, Not Interactions:**
   - ❌ BAD: "User can click the save button."
   - ✅ GOOD: "Persist changes to the user profile."

2. **IGNORE Global & Common Elements:**
   - **CRITICAL:** Do NOT list use cases for the Sidebar, Navigation Menu, Header, Footer, or Chat Widget.
   - Focus ONLY on the unique content of this specific page.

3. **Output Format:**
   - You MUST return a single valid JSON object.
   - Keys: "description" (string), "usecases" (list of strings).
   - Do not include markdown formatting (like ```json). Just the raw JSON object.

### FEW-SHOT EXAMPLES
{few_shot_examples}
"""

USER_PROMPT_TEMPLATE = """
### COMPONENT FILE: {component_path}
### CODE CONTENT:
{file_content}
"""

class EnrichmentAgent:
    def __init__(self, model_name="gpt-4o"):
        # We force JSON mode to prevent parsing errors
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.parser = JsonOutputParser(pydantic_object=RouteEnrichment)
        
        self.few_shots = self._load_few_shots()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def _load_few_shots(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "templates", "fewshots.json")
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: Few-shots file not found at {path}. Using empty context.")
            return "[]"
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ Error reading few-shots: {e}")
            return "[]"

    async def analyze_component_with_retry(self, route_entry: Dict, retries=3) -> Dict:
        """
        Analyzes a single component with retry logic for Rate Limits.
        """
        component_path = route_entry.get("component")
        base_dir = os.path.join(os.getcwd(), "bilkent-tanitim", "frontend")
        full_path = os.path.join(base_dir, component_path)
        
        if not os.path.exists(full_path):
            print(f"   ⚠️  File not found: {full_path}")
            route_entry["description"] = "Error: File not found."
            route_entry["usecases"] = []
            return route_entry

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            route_entry["description"] = f"Error reading file: {str(e)}"
            return route_entry

        print(f"   🔍 Analyzing {component_path}...")
        
        for attempt in range(retries):
            try:
                result = await self.chain.ainvoke({
                    "component_path": component_path,
                    "file_content": content,
                    "few_shot_examples": self.few_shots
                })
                
                # Merge results
                route_entry["description"] = result.get("description", "No description.")
                route_entry["usecases"] = result.get("usecases", [])
                return route_entry

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                    print(f"      ⏳ Rate Limit hit. Waiting {wait_time}s before retry ({attempt+1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ Parse/LLM Error: {e}")
                    # If it's not a rate limit, it might be a parsing error.
                    # We return empty to skip this file and move on.
                    route_entry["description"] = "Error during analysis."
                    route_entry["usecases"] = []
                    return route_entry
        
        return route_entry

async def run_enrichment(routes_path="routes.json"):
    print(f"🚀 Starting Route Enrichment (Incremental Save)...")
    
    if not os.path.exists(routes_path):
        print("❌ routes.json not found.")
        return

    # Load initial state
    with open(routes_path, "r", encoding="utf-8") as f:
        routes = json.load(f)

    agent = EnrichmentAgent()

    # Iterate through routes
    for i in range(len(routes)):
        route = routes[i]
        
        # Skip if already enriched (optional check, currently disabled to force update)
        # if "description" in route and len(route.get("usecases", [])) > 0:
        #     continue

        print(f"[{i+1}/{len(routes)}] Processing {route.get('path')}...")
        
        # Process single route
        updated_route = await agent.analyze_component_with_retry(route)
        routes[i] = updated_route # Update the list in memory
        
        # INCREMENTAL SAVE: Write to disk immediately
        try:
            with open(routes_path, "w", encoding="utf-8") as f:
                json.dump(routes, f, indent=2)
            # print(f"      💾 Saved progress.")
        except Exception as save_error:
            print(f"      ❌ Error saving to disk: {save_error}")

        # Rate Limit Buffer: Sleep 1 second between requests to be kind to the API
        time.sleep(1)

    print(f"✅ Enrichment Complete! All routes updated.")

if __name__ == "__main__":
    asyncio.run(run_enrichment())