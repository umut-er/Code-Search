import json
import os
from typing import Dict, List

from langchain_core.tools import tool

from enhancement.src.bug_report_enhancer import BugReportEnhancer
from enhancement.src.reproduction_generator import ReproductionGenerator


@tool("generate_s2r")
def generate_s2r(bug_report: str, code_paths: str) -> str:
    """
    Generate a browser-ready Steps-to-Reproduce (S2R) JSON object.

    Args:
        bug_report: The raw, badly written bug report text supplied by the user.
        code_paths: A JSON array or comma-separated list of file paths
                    (TypeScript / HTML / CSS) that are most relevant to this bug.
                    Example:
                      '["src/pages/Checkout.tsx", "src/components/CartSummary.tsx"]'
                      or
                      "src/pages/Checkout.tsx, src/components/CartSummary.tsx"

    Behavior:
        1. Uses the BugReportEnhancer module to extract OB / EB / S2R from the report.
        2. Reads the specified code files into memory as context.
        3. Calls the ReproductionGenerator to produce a structured JSON with:
           - "analysis": short reasoning
           - "steps":    ordered list of low-level browser actions

    Returns:
        A pretty-printed JSON string representing the S2R automation script,
        suitable for handing off to a downstream browser agent.
    """
    # --- 1. Normalize and parse code_paths into a list of strings ---
    paths: List[str] = []
    raw = (code_paths or "").strip()

    if not raw:
        # The agent forgot to provide paths – fail fast with instructions.
        return json.dumps(
            {
                "error": "NO_CODE_PATHS_PROVIDED",
                "message": (
                    "generate_s2r requires at least one relevant code path. "
                    "Use your code navigation tools (find_file, grep_text, read_file, etc.) "
                    "to locate the most relevant TypeScript/HTML/CSS files and pass them in."
                ),
            },
            indent=2,
        )

    # Accept either a JSON array or a simple comma-separated string.
    try:
        if raw.startswith("["):
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                paths = [str(p).strip() for p in parsed if str(p).strip()]
        else:
            paths = [p.strip() for p in raw.split(",") if p.strip()]
    except Exception:
        # Fallback: naive comma-split
        paths = [p.strip() for p in raw.split(",") if p.strip()]

    # --- 2. Load code files into a {filename: content} mapping ---
    code_files: Dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            # Skip missing files but keep going so one bad path doesn't kill the run.
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                code_files[path] = f.read()
        except Exception:
            # Best-effort read – if a file fails, we just skip it.
            continue

    if not code_files:
        return json.dumps(
            {
                "error": "NO_CODE_CONTENT",
                "message": (
                    "None of the provided code_paths could be read. "
                    "Verify that the paths exist on disk and are readable."
                ),
                "received_paths": paths,
            },
            indent=2,
        )

    # --- 3. Enhance the bug report into structured OB / EB / S2R ---
    enhancer = BugReportEnhancer()
    structured_bug = enhancer.extract_fields(bug_report)

    # --- 4. Generate the automation-ready S2R JSON ---
    generator = ReproductionGenerator()
    s2r_payload = generator.generate_steps(structured_bug, code_files)

    # Ensure we always return a JSON string, even on failure.
    if s2r_payload is None:
        return json.dumps(
            {
                "error": "GENERATION_FAILED",
                "message": "ReproductionGenerator returned None. Check logs for details.",
            },
            indent=2,
        )

    if isinstance(s2r_payload, (dict, list)):
        return json.dumps(s2r_payload, indent=2)

    # Fallback: return raw string content.
    return json.dumps({"raw": str(s2r_payload)}, indent=2)


