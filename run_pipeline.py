import os
import json
import asyncio
import logging
import time
from dotenv import load_dotenv
from openai import RateLimitError

# --- IMPORTS ---
# 1. Jira Integration
from enhancement.src.fetch_jira import fetch_bug_reports
# 2. Enhancement Agent
from enhancement.src.bug_report_enhancer import BugReportEnhancer
# 3. Search Agent
from SearchAgentEngine import SearchAgentEngine

from enhancement.src.reproduction_generator import ReproductionGenerator

# Load Environment Variables
load_dotenv()

# Configure Pipeline Logging
logging.basicConfig(level=logging.INFO, format='[PIPELINE] %(message)s')
logger = logging.getLogger(__name__)

class BugFixPipeline:
    """
    Orchestrator class that manages the flow of data between different agents.
    Current Flow: Jira -> Enhancer -> Search Agent -> (Output: final_context.json)
    Future Flow: ... -> Search Agent -> S2R Agent -> Browser Agent
    """
    def __init__(self):
        self.enhancer_agent = BugReportEnhancer(model_name="gpt-4o")
        self.search_engine = SearchAgentEngine()
        self.reproduction_generator = ReproductionGenerator()
    async def process_latest_ticket(self):
        """
        Fetches the latest bug from Jira and runs the full diagnosis pipeline.
        """
        logger.info("🌊 STARING PIPELINE...")

        # ====================================================
        # PHASE 1: FETCH (Jira Ingestion)
        # ====================================================
        logger.info("--- PHASE 1: FETCHING FROM JIRA ---")
        tickets = fetch_bug_reports()
        
        if not tickets:
            logger.warning("❌ No bug reports found in Jira. Exiting pipeline.")
            return

        # For this implementation, we process the most recent ticket
        latest_ticket = tickets[0]
        ticket_id = latest_ticket.get('key')
        logger.info(f"✅ Processing Ticket: {ticket_id} | {latest_ticket.get('summary')}")

        # ====================================================
        # PHASE 2: ENHANCE (Structured Data Extraction)
        # ====================================================
        logger.info("--- PHASE 2: ENHANCING BUG REPORT ---")
        
        # Combine Summary and Description for the Enhancer
        raw_text = f"Title: {latest_ticket.get('summary')}\nDescription: {latest_ticket.get('description')}"
        
        # The Enhancer Agent structures this into {ID, Title, OB, EB, S2R}
        structured_bug = self.enhancer_agent.extract_fields(raw_text)
        
        print(structured_bug)
        # Ensure the ID from Jira is preserved even if LLM missed it
        if isinstance(structured_bug, dict):
            structured_bug['ID'] = ticket_id
        else:
            # Fallback if enhancer failed to return dict
            logger.error("❌ Enhancer failed to return structured JSON.")
            return

        logger.info(f"✅ Report Enhanced. S2R detected: {bool(structured_bug.get('S2R'))}")
        # print(json.dumps(structured_bug, indent=2)) # Debug print

        # ====================================================
        # PHASE 3: SEARCH (Code Context Retrieval)
        # ====================================================
        logger.info("--- PHASE 3: SEARCH AGENT EXECUTION ---")
        
        # We construct a prompt that effectively hands off the structured data to the Search Agent.
        # This acts as the "Context Injection" for the search engine.
        agent_handshake_prompt = f"""
        INCOMING ASSIGNMENT: BUG INVESTIGATION
        
        BUG INFO:
        {json.dumps(structured_bug, indent=2)}

        YOUR MISSION:
        1. Analyze the 'S2R' (Steps to Reproduce) and 'OB' (Observed Behavior) above.
        2. Use `find_best_route` to determine where to start in the codebase.
        3. Explore the project to identify the specific files causing this issue.
        4. CRITICAL: You must call `submit_final_report` when you have found the files. 
           Do not stop until you have called this tool.
        """

        # Execute the Search Agent
        # We use the Jira Ticket ID as the thread_id to keep memory isolated per ticket
        result = await self.search_engine.process_request(
            message=agent_handshake_prompt, 
            thread_id=ticket_id
        )

        if result['status'] != 'success':
            logger.error(f"❌ Search Agent Failed: {result.get('error')}")
            return

        # ====================================================
        # PHASE 4: VERIFICATION & HANDOFF (Future S2R Integration)
        # ====================================================
        logger.info("--- PHASE 4: VERIFICATION ---")
        
        # Check if the agent successfully produced the artifact defined in local_tools.py
        if os.path.exists("final_context.json"):
            logger.info("🎉 SUCCESS: 'final_context.json' generated successfully.")
            
            try:
                # 1. Dosyayı Oku
                with open("final_context.json", "r", encoding="utf-8") as f:
                    final_data = json.load(f)

                # 2. Bug Datasini Ayıkla
                # Generator sadece Title, OB, EB ve S2R'a ihtiyaç duyar
                bug_payload = {
                    "Title": final_data.get("Title"),
                    "OB": final_data.get("OB"),
                    "EB": final_data.get("EB"),
                    "S2R": final_data.get("S2R")
                }

                # 3. Code Dosyalarini Dictionary Formatina Çevir
                # final_context.json içinde liste halindeler, bunu {filename: content} formatına çevirmeliyiz
                code_files_payload = {}
                relevant_files_list = final_data.get("relevant_files", [])
                
                if not relevant_files_list:
                    logger.warning("⚠️ No relevant files found in final_context.json. S2R might be poor.")

                for file_obj in relevant_files_list:
                    # Dosya adını ve içeriğini al
                    fname = file_obj.get("name") or os.path.basename(file_obj.get("path", "unknown"))
                    content = file_obj.get("content", "")
                    code_files_payload[fname] = content

                # 4. Starting Route'u Al
                # final_context.json içinde root seviyesinde olmalı
                starting_route = final_data.get("starting_route", "/") 
                
                logger.info(f"🚀 Generatig S2R for route: {starting_route} with {len(code_files_payload)} files context.")

                # 5. GENERATOR ÇAĞRISI (with rate limit retry)
                max_retries = 3
                s2r_output = None
                
                for attempt in range(max_retries):
                    try:
                        s2r_output = self.reproduction_generator.generate_steps(
                            bug_data=bug_payload,
                            code_files=code_files_payload,
                            starting_route=starting_route
                        )
                        break  # Success, exit retry loop
                    except RateLimitError as e:
                        if attempt < max_retries - 1:
                            wait_time = 20
                            logger.warning(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ Rate limit error after {max_retries} attempts.")
                            raise
                    except Exception as e:
                        # Non-rate-limit errors: re-raise immediately
                        logger.error(f"❌ Error during S2R generation: {e}")
                        raise

                # 6. Sonucu Kaydet (s2r.json)
                if s2r_output:
                    output_filename = "s2r.json"
                    with open(output_filename, "w", encoding="utf-8") as out_f:
                        json.dump(s2r_output, out_f, indent=2)
                    
                    logger.info(f"✅ SUCCESS: Steps to Reproduce saved to '{output_filename}'")
                else:
                    logger.error("❌ S2R Generation returned None.")

            except Exception as e:
                logger.error(f"❌ Error during S2R Generation Phase: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("⚠️  'final_context.json' not found. Cannot proceed to S2R generation.")

if __name__ == "__main__":
    pipeline = BugFixPipeline()
    asyncio.run(pipeline.process_latest_ticket())