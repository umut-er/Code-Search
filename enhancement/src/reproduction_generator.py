import json
import os
from openai import OpenAI

class ReproductionGenerator:
    """
    Takes a bug report (string or structured dict) and code context, 
    then generates automation steps using OpenAI.
    """
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        
        # 1. Initialize OpenAI Client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        self.client = OpenAI(api_key=api_key)

        # 2. Load the external Prompt Template
        # This logic finds 'templates/reproduction_prompt.txt' relative to this script's location
        base_dir = os.path.dirname(__file__)  # src/
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        template_path = os.path.join(project_root, "templates", "reproduction_prompt.txt")
        
        self.prompt_template = self._load_text_file(template_path)

    @staticmethod
    def _load_text_file(path: str) -> str:
        """Helper to safely read the text file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _construct_prompt(self, bug_data, code_files):
        """
        Formats the inputs and injects them into the text template.
        """
        # A. Format the Code Files
        code_context_str = ""
        for filename, content in code_files.items():
            code_context_str += f"--- FILE: {filename} ---\n{content}\n\n"

        # B. Format the Bug Description (handles both Dict and String input)
        if isinstance(bug_data, dict):
            # If input is structured JSON (from the Enhancer module)
            bug_context_str = f"""
- **Title**: {bug_data.get('Title', 'N/A')}
- **Observed Behavior (OB)**: {bug_data.get('OB', 'N/A')}
- **Expected Behavior (EB)**: {bug_data.get('EB', 'N/A')}
- **Steps to Reproduce (S2R)**: {bug_data.get('S2R', 'N/A')}
            """.strip()
        else:
            # If input is just a raw string (fallback)
            bug_context_str = str(bug_data)

        # C. Fill the Template placeholders
        # {BUG_CONTEXT} and {CODE_CONTEXT} match the keys in reproduction_prompt.txt
        return self.prompt_template.format(
            BUG_CONTEXT=bug_context_str,
            CODE_CONTEXT=code_context_str
        )

    def generate_steps(self, bug_data, code_files):
        """
        Main method to generate the steps.
        """
        print(f"AGENT: Generating automation steps using {self.model_name}...")
        prompt = self._construct_prompt(bug_data, code_files)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a QA Automation Expert. Output strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Sanitize markdown if the model wraps it in ```json ... ```
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "")
            elif raw_content.startswith("```"):
                 raw_content = raw_content.replace("```", "")
            
            return json.loads(raw_content)

        except Exception as e:
            print(f" [Generator] ERROR: {e}")
            return None