import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Import the agents
from src.bug_report_enhancer import BugReportEnhancer
from src.reproduction_generator import ReproductionGenerator
from src.fetch_jira import fetch_bug_reports
from src.fetch_jira import add_comment_to_issue
from src.fetch_jira import transition_issue_to

load_dotenv()
used_model_name = "gpt-4o"

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "log")
    return safe[:80] or "log"

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _write_log_text(prefix: str, suffix: str, content: str) -> str:
    _ensure_logs_dir()
    filename = f"{_sanitize_filename(prefix)}_{_timestamp()}_{suffix}"
    path = os.path.join(LOGS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content is not None else "")
    return path

def _write_log_json(prefix: str, suffix: str, data) -> str:
    _ensure_logs_dir()
    filename = f"{_sanitize_filename(prefix)}_{_timestamp()}_{suffix}"
    path = os.path.join(LOGS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def process_jira_reports():
    """
    Fetch latest bug reports from Jira and run the pipeline on them.
    Replaces reading from local examples.
    """
    # 1. Initialize Agents
    try:
        enhancer = BugReportEnhancer(model_name=used_model_name, temperature=0.0)
        generator = ReproductionGenerator(model_name=used_model_name)
    except Exception as e:
        return

    # 2. Fetch from Jira
    bugs = fetch_bug_reports()

    if not bugs:
        return

    # 3. Take the most recent bug (first item; JQL orders by created DESC), later
    latest = bugs[0]
    bug_key = latest.get("key", "UNKNOWN")
    summary = latest.get("summary") or ""
    description = latest.get("description") or ""

    # Compose a readable text for the enhancer
    raw_bug_text = f"Title: {summary}\n\nDescription:\n{description}".strip()
    _write_log_text(bug_key, "raw.txt", raw_bug_text)

    # 4. Run Enhancer
    structured_bug = None
    try:
        structured_bug = enhancer.extract_fields(raw_bug_text)
    except Exception as e:
        return

    if not structured_bug:
        return

    _write_log_json(bug_key, "enhanced.json", structured_bug)

    # 5. Generate Reproduction Steps (no code context by default)
    # DEMO: Load mock retrieval context from examples_context_retrieval/mid_report_context.txt
    code_context = {}
    try:
        base_dir = os.path.dirname(__file__)
        mock_context_path = os.path.join(base_dir, "examples_context_retrieval", "mid_report_context.txt")
        if os.path.exists(mock_context_path):
            with open(mock_context_path, "r", encoding="utf-8") as f:
                mock_context_content = f.read()
                code_context = {"mid_report_context.txt": mock_context_content}
    except Exception:
        code_context = {}
    automation_result = generator.generate_steps(structured_bug, code_context)

    if automation_result:
        _write_log_json(bug_key, "automation.json", automation_result)
        # Post automation output to Jira as a comment
        try:
            enhanced_str = json.dumps(structured_bug, indent=2, ensure_ascii=False)
            analysis = automation_result.get("analysis")
            steps = automation_result.get("steps") or []
            steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            parts = []
            parts.append("Enhanced bug report (auto-generated):")
            parts.append(enhanced_str)
            parts.append("\nAutomation steps (auto-generated):")
            if analysis:
                parts.append("\nAnalysis:")
                parts.append(analysis)
            if steps:
                parts.append("\nSteps:")
                parts.append(steps_text)
            comment_text = "\n".join(parts)
            add_comment_to_issue(bug_key, comment_text)
            # Move issue status to In Progress
            transition_issue_to(bug_key, "In Progress")
        except Exception:
            pass
    else:
        pass

if __name__ == "__main__":
    process_jira_reports()