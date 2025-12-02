import os
import subprocess
import json

from langchain_core.tools import StructuredTool, tool

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

class LocalTools:
    """
    Native Python tools for File I/O and Lexical Search.
    This replaces the buggy Filesystem MCP.
    """

    @tool("get_code_symbols")
    def get_code_symbols(path: str) -> str:
        """
        Parses a file using Tree-sitter (AST) to find classes, functions, and methods.
        Returns the structure with EXACT line numbers [start-end].
        Supports: Python, TypeScript, JS, Go, Rust, C++.
        """
        if not os.path.exists(path):
            return f"Error: File not found at {path}"

        ext = os.path.splitext(path)[1].lower()
        lang_map = {
            ".py": "python", ".ts": "typescript", ".tsx": "tsx",
            ".js": "javascript", ".jsx": "javascript", 
            ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp"
        }
        
        lang_name = lang_map.get(ext)
        if not lang_name:
            return f"Error: Unsupported file extension {ext} for AST parsing."

        try:
            parser = get_parser(lang_name)
            language = get_language(lang_name)
            
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = parser.parse(bytes(code, "utf8"))
            
            # Standard S-Expression for definitions
            query_scm = """
            (class_declaration name: (_) @name) @def
            (function_declaration name: (_) @name) @def
            (interface_declaration name: (_) @name) @def
            (method_definition name: (_) @name) @def
            """
            
            if lang_name == "python":
                query_scm = """
                (class_definition name: (_) @name) @def
                (function_definition name: (_) @name) @def
                """
            elif lang_name == "go":
                query_scm = """
                (function_declaration name: (_) @name) @def
                (method_declaration name: (_) @name) @def
                """

            # --- NEW API IMPLEMENTATION ---
            query = Query(language, query_scm)
            cursor = QueryCursor(query)
            captures_obj = cursor.captures(tree.root_node)
            
            results = []
            seen_lines = set()
            
            # Flatten results if dict (v0.22+ behavior)
            all_captures = []
            if isinstance(captures_obj, dict):
                for name, nodes in captures_obj.items():
                    for node in nodes:
                        all_captures.append((node, name))
            else:
                all_captures = captures_obj

            for node, capture_name in all_captures:
                if capture_name == "def":
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    
                    if start_line in seen_lines: continue
                    seen_lines.add(start_line)

                    lines = code.splitlines()
                    if start_line - 1 < len(lines):
                        signature = lines[start_line-1].strip()
                        if len(signature) > 80: 
                            signature = signature[:77] + "..."
                        
                        kind = node.type.replace("_declaration", "").replace("_definition", "")
                        results.append(f"[{start_line}-{end_line}] {kind.upper()}: {signature}")

            if not results:
                return "No top-level definitions found."

            # Sort by line number
            results.sort(key=lambda x: int(x.split('-')[0].replace('[', '')))
            return "\n".join(results)

        except Exception as e:
            import traceback
            return f"AST Parsing Error: {e}\n{traceback.format_exc()}"

    @tool("read_file")
    def read_file(path: str, start_line: int = 1, end_line: int = -1) -> str:
        """
        Read the content of a file. You can read specific line ranges to save tokens.
        
        Args:
            path: The relative path to the file.
            start_line: The first line to read (1-based index). Default is 1.
            end_line: The last line to read (1-based index). Default is -1 (Read until end of file).
                      Example: start_line=10, end_line=20 will read lines 10 through 20.
        """
        try:
            if not os.path.exists(path):
                return f"Error: File not found at {path}"
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Handle end_line = -1 (read to end)
            if end_line == -1:
                end_line = total_lines
                
            # Clamp values to valid ranges
            if start_line < 1: 
                start_line = 1
            if end_line > total_lines: 
                end_line = total_lines
            
            if start_line > end_line:
                return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line}). Total lines: {total_lines}"

            # Python lists are 0-indexed, so we adjust start_line
            # Slicing is exclusive at the end, so end_line is actually correct as-is for the upper bound
            selected_lines = lines[start_line-1 : end_line]
            
            content = "".join(selected_lines)
            
            # Return with metadata so the LLM understands the context
            return (
                f"--- Reading {path} (Lines {start_line}-{end_line} of {total_lines}) ---\n"
                f"{content}\n"
                f"--- End of Snippet ---"
            )

        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool("list_directory")
    def list_directory(path: str = ".") -> str:
        """
        List files and subdirectories in a given path.
        Useful for exploring the codebase structure.
        """
        try:
            if not os.path.isdir(path):
                return f"Error: {path} is not a directory."
            
            items = os.listdir(path)
            formatted = []
            for item in items:
                # Skip hidden files like .git to save noise
                if item.startswith("."): 
                    continue
                    
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    formatted.append(f"[DIR]  {item}")
                else:
                    formatted.append(f"[FILE] {item}")
            return "\n".join(sorted(formatted))
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @tool("search_code_lexical")
    def search_code(query: str, path: str = ".", context_lines: int = 1) -> str:
        """
        Fast lexical search using Ripgrep (rg). Finds exact strings or regex patterns.
        Use this to find where variables, error messages, or functions are defined.
        
        Args:
            query: The regex pattern or text to search for.
            path: The directory to search in (defaults to current dir).
            context_lines: Number of lines to show around the match (default 1).
        """
        try:
            # Check for ripgrep
            if subprocess.call(["which", "rg"], stdout=subprocess.DEVNULL) != 0:
                return "Error: 'rg' (ripgrep) is not installed on this system."

            command = [
                "rg", 
                "-n", # Line numbers
                f"-C{context_lines}", # Context
                "--json", # JSON output for reliable parsing
                "-e", query, 
                path
            ]
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 1:
                return "No matches found."
            elif result.returncode > 1:
                return f"Ripgrep Error: {result.stderr}"

            matches = []
            for line in result.stdout.splitlines():
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        file_path = data["data"]["path"]["text"]
                        line_num = data["data"]["line_number"]
                        line_text = data["data"]["lines"]["text"].strip()
                        matches.append(f"{file_path}:{line_num} | {line_text}")
                except:
                    continue
            
            if not matches:
                return "No matches found."
            
            # Limit to 10 matches to prevent token overflow on common searches
            return "\n".join(matches[:10])
            
        except Exception as e:
            return f"Search execution error: {str(e)}"
