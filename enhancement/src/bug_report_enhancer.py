import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (e.g., API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) 


class BugReportEnhancer:
    """
    Enhance bug reports by:
      - detecting Observed Behavior (OB), Expected Behavior (EB), and Steps to Reproduce (S2R)
      - evaluating the quality of each field
      - returning a structured JSON object

    The core prompt lives in templates/prompt.txt and is filled with:
      - the output schema (templates/schema.json)
      - the few-shot examples block (templates/fewshots.txt)
      - the new bug report text
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

        # Resolve project paths
        base_dir = os.path.dirname(__file__)  # src/
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        templates_dir = os.path.join(project_root, "templates")

        self.prompt_template_path = os.path.join(templates_dir, "prompt.txt")
        self.schema_path = os.path.join(templates_dir, "schema.json")
        self.fewshots_path = os.path.join(templates_dir, "fewshots.txt")

        # Load static resources once
        self.prompt_template = self._load_text_file(self.prompt_template_path)
        self.schema_json = self._load_text_file(self.schema_path)
        self.fewshots_block = self._load_text_file(self.fewshots_path)

    @staticmethod
    def _load_text_file(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _compose_prompt(self, new_report_text: str) -> str:
        """
        Fill the prompt template with:
          - OUTPUT_SCHEMA_JSON
          - FEWSHOTS_BLOCK
          - BUG_REPORT_TEXT
        The template is responsible for instructions + formatting.
        """
        return self.prompt_template.format(
            OUTPUT_SCHEMA_JSON=self.schema_json,
            FEWSHOTS_BLOCK=self.fewshots_block,
            BUG_REPORT_TEXT=new_report_text.strip(),
        )

    def extract_fields(self, bug_report_text: str):
        """
        Call the LLM to extract structured fields from a bug report.
        Returns:
          - a Python dict parsed from the JSON response, or
          - raw text if JSON parsing fails (with a warning).
        """
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        prompt = self._compose_prompt(bug_report_text)

        try:
            response = client.chat.completions.create(
                model=self.model_name,  # e.g. "gpt-4.1-mini" or "gpt-4o-mini"
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only bug report extractor. Always return a single JSON object."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {e}")

        output_text = response.choices[0].message.content.strip()

        json_match = re.search(r"```json\s*(.*?)\s*```", output_text, re.DOTALL)
        if json_match:
            output_text = json_match.group(1)
        elif output_text.startswith("```"):
            output_text = output_text.strip("`").strip()

        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON. Returning raw output.")
            return output_text
