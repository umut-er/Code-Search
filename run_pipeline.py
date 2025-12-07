import os
import json
import asyncio
import logging
from dotenv import load_dotenv

# --- IMPORTS ---
# 1. Jira Integration
from enhancement.src.fetch_jira import fetch_bug_reports
# 2. Enhancement Agent
from enhancement.src.bug_report_enhancer import BugReportEnhancer
# 3. Search Agent
from SearchAgentEngine import SearchAgentEngine

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
            
            with open("final_context.json", "r") as f:
                final_data = json.load(f)
                file_count = len(final_data.get("relevant_files", []))
                logger.info(f"📁 Agent identified {file_count} relevant files for reproduction.")
            
            # ----------------------------------------------------------------
            # FUTURE INTEGRATION POINT:
            # Here is where we will call the S2R Generator Agent.
            # Example:
            # s2r_agent.generate_script(
            #     bug_json=structured_bug, 
            #     code_context=final_data['relevant_files']
            # )
            # ----------------------------------------------------------------
        else:
            logger.warning("⚠️  Pipeline finished, but 'final_context.json' was not found.")
            logger.warning("The agent may have chatted but failed to call 'submit_final_report'.")

if __name__ == "__main__":
    pipeline = BugFixPipeline()
    asyncio.run(pipeline.process_latest_ticket())