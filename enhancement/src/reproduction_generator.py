import json
import os
from openai import OpenAI, RateLimitError

class ReproductionGenerator:
    """
    Harmonizes user reports with code findings to generate browser-use friendly steps.
    """
    def __init__(self, model_name="gpt-4o"): # Default to gpt-4o for best "Mental Rendering"
        self.model_name = model_name
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        self.client = OpenAI(api_key=api_key)

        # Load the new prompt template
        base_dir = os.path.dirname(__file__) 
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        # Ensure you updated the text file at this path with the content above
        template_path = os.path.join(project_root, "templates", "reproduction_prompt.txt")
        
        self.prompt_template = self._load_text_file(template_path)

    @staticmethod
    def _load_text_file(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _construct_prompt(self, bug_data, code_files, starting_route):
        # Format Code Context
        code_context_str = ""
        for filename, content in code_files.items():
            code_context_str += f"--- FILE: {filename} ---\n{content}\n\n"

        # Format Bug Data
        if isinstance(bug_data, dict):
            bug_context_str = f"""
            - Title: {bug_data.get('Title', 'N/A')}
            - Observed Behavior: {bug_data.get('OB', 'N/A')}
            - Steps to Reproduce: {bug_data.get('S2R', 'N/A')}
            """.strip()
        else:
            bug_context_str = str(bug_data)

        return self.prompt_template.format(
            BUG_CONTEXT=bug_context_str,
            CODE_CONTEXT=code_context_str,
            STARTING_ROUTE=starting_route
        )

    def generate_steps(self, bug_data, code_files, starting_route):
        print(f"   🧠 Harmonizing User Intent with Code Reality using {self.model_name}...")
        prompt = self._construct_prompt(bug_data, code_files, starting_route)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating Natural Language instructions for browser-use agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Low temperature for consistent adherence to code facts
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Basic cleanup
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "")
            
            return json.loads(raw_content)

        except RateLimitError as e:
            # Re-raise rate limit errors so caller can retry
            print(f" [Generator] Rate Limit Error: {e}")
            raise
        except Exception as e:
            print(f" [Generator] ERROR: {e}")
            return None