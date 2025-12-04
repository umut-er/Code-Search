import os
import subprocess
import json
import difflib

from langchain_core.tools import StructuredTool, tool

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

class LocalTools:
    """
    Native Python tools for File I/O and Lexical Search.
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

    @tool("find_file")
    def find_file(name_pattern: str, path: str = ".") -> str:
        """
        Fuzzy searches for files by filename (not content). 
        Useful when you know the approximate name of a file (e.g., 'auth' -> 'Authentication.ts').
        
        Args:
            name_pattern: The partial name or fuzzy guess of the file.
            path: Root directory to start search (default is current).
        """
        try:
            results = []
            # Common web dev folders to ignore to speed up search
            exclude_dirs = {
                "node_modules", ".git", ".next", "dist", "build", 
                "coverage", ".vscode", "__pycache__"
            }
            
            # 1. First pass: exact substring matching (fastest/most accurate)
            # 2. Second pass: fuzzy matching if substring fails
            
            candidates = []
            
            for root, dirs, files in os.walk(path):
                # Modify dirs in-place to skip ignored directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    full_path = os.path.join(root, file)
                    candidates.append(full_path)
                    
                    # Case-insensitive substring match
                    if name_pattern.lower() in file.lower():
                        results.append(full_path)

            # If we found exact substring matches, return those (limit 10)
            if results:
                results.sort(key=len) # Shortest paths first often meant "source" vs "compiled"
                return "\n".join(results[:10])
            
            # If no substring match, try fuzzy matching on filenames
            # This helps if the agent types "login_modal" but file is "LoginModal.tsx"
            filenames = [os.path.basename(c) for c in candidates]
            close_matches = difflib.get_close_matches(name_pattern, filenames, n=5, cutoff=0.6)
            
            fuzzy_results = []
            if close_matches:
                for match in close_matches:
                    # Find the full path for the matched filename
                    for c in candidates:
                        if os.path.basename(c) == match:
                            fuzzy_results.append(c)
            
            if not fuzzy_results:
                return f"No files found matching '{name_pattern}'."
            
            return "No exact matches. Did you mean:\n" + "\n".join(fuzzy_results[:5])

        except Exception as e:
            return f"Error finding file: {str(e)}"

    @tool("find_usage")
    def find_usage(filename: str, path: str = ".") -> str:
        """
        Finds where a specific file or component is used in the codebase.
        This is useful for tracing a component upwards to find the Page or URL that renders it.
        
        Args:
            filename: The name of the file/component to find usages for (e.g., "SubmitButton.tsx" or just "SubmitButton").
            path: The root directory to search in.
        """
        try:
            # Check for ripgrep
            if subprocess.call(["which", "rg"], stdout=subprocess.DEVNULL) != 0:
                return "Error: 'rg' (ripgrep) is not installed."

            # 1. Normalize the name. 
            # If input is "SubmitButton.tsx", we want to search for "SubmitButton"
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # 2. Construct specific Regex patterns for TypeScript/Web usage
            # Pattern A: Import statements (e.g., import { X } from './X')
            # Pattern B: JSX Usage (e.g., <X /> or <X)
            # We combine them with OR (|)
            # We look for the filename in the import path OR the component name in JSX tags
            pattern = f"from ['\"].*{base_name}['\"]|<{base_name}(\\s|/|>)"
            
            command = [
                "rg", 
                "-n", 
                "--json", 
                "-e", pattern, 
                path,
                "-g", "!node_modules", # Ignore node_modules
                "-g", "!dist",         # Ignore build output
                "-g", "!build"
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            matches = []
            seen_files = set()

            for line in result.stdout.splitlines():
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        file_path = data["data"]["path"]["text"]
                        
                        # Don't list the file itself as a usage of itself
                        if os.path.basename(file_path) == os.path.basename(filename):
                            continue
                            
                        line_num = data["data"]["line_number"]
                        line_text = data["data"]["lines"]["text"].strip()
                        
                        # To keep context manageable, we try to group by file if there are many matches
                        matches.append(f"{file_path}:{line_num} | {line_text}")
                        seen_files.add(file_path)
                except:
                    continue

            if not matches:
                return f"No usages found for '{base_name}'. It might be unused or dynamically imported."

            # Formatting logic: If too many matches, summarize
            if len(matches) > 15:
                unique_files_list = "\n".join(list(seen_files)[:10])
                return (f"Found {len(matches)} usages in {len(seen_files)} files. "
                        f"Here are the files where it appears:\n{unique_files_list}\n"
                        f"(Use read_file on these to see specific implementation)")
            
            return "\n".join(matches)

        except Exception as e:
            return f"Error finding usage: {str(e)}"

    @tool("read_file_skeleton")
    def read_file_skeleton(path: str) -> str:
        """
        Reads a file and returns a skeleton view by collapsing function/class bodies.
        Useful for understanding high-level file structure without reading every line of code.
        
        Args:
            path: Relative path to the file.
        """
        if not os.path.exists(path):
            return f"Error: File not found at {path}"

        # 1. Detect Language
        ext = os.path.splitext(path)[1].lower()
        lang_map = {
            ".py": "python", ".ts": "typescript", ".tsx": "tsx",
            ".js": "javascript", ".jsx": "javascript", 
            ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp"
        }
        lang_name = lang_map.get(ext)
        
        # Fallback: simple read if language not supported
        if not lang_name:
             return LocalTools.read_file(path, end_line=100) + "\n... (Unsupported language for skeleton, showing first 100 lines)"

        try:
            parser = get_parser(lang_name)
            language = get_language(lang_name)
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            
            tree = parser.parse(bytes(code, "utf8"))
            
            # 2. Define queries to find foldable blocks
            # We target function bodies, class bodies, and method bodies
            if lang_name in ["typescript", "tsx", "javascript"]:
                query_scm = """
                (function_declaration body: (_) @body)
                (method_definition body: (_) @body)
                (arrow_function body: (_) @body)
                (class_declaration body: (_) @body)
                """
            elif lang_name == "python":
                query_scm = """
                (function_definition body: (_) @body)
                (class_definition body: (_) @body)
                """
            elif lang_name == "go":
                query_scm = """
                (function_declaration body: (_) @body)
                (method_declaration body: (_) @body)
                """
            elif lang_name == "rust":
                query_scm = """
                (function_item body: (_) @body)
                (impl_item body: (_) @body)
                """
            else:
                query_scm = "(function_definition body: (_) @body)"

            # 3. Execute Query
            query = Query(language, query_scm)
            cursor = QueryCursor(query)
            captures_obj = cursor.captures(tree.root_node)
            
            # Normalize captures (v0.22+ returns dict, older returns list)
            nodes = []
            if isinstance(captures_obj, dict):
                for name, captured_nodes in captures_obj.items():
                    nodes.extend(captured_nodes)
            else:
                nodes = [node for node, name in captures_obj]

            # 4. Filter and Sort Folds
            # We sort by start line to handle nesting logic
            nodes.sort(key=lambda n: n.start_point[0])
            
            fold_ranges = []
            last_fold_end_line = -1
            
            for node in nodes:
                start_line = node.start_point[0] # 0-indexed
                end_line = node.end_point[0]     # 0-indexed
                
                # Heuristic: Only fold if body is > 4 lines long
                if (end_line - start_line) <= 4:
                    continue

                # NESTING FIX: 
                # If this new node starts BEFORE the last folded node ended, 
                # it is a child/nested node. We skip it to keep the parent folded.
                if start_line <= last_fold_end_line:
                    continue
                
                # We fold from start+1 to end-1 (keeping the opening/closing braces visible)
                fold_start = start_line + 1
                fold_end = end_line - 1
                
                if fold_start <= fold_end:
                    fold_ranges.append((fold_start, fold_end))
                    last_fold_end_line = end_line # Update boundary

            # 5. Create a map for O(1) lookups during reconstruction
            # Map: line_index -> (is_start_of_fold, fold_length)
            hidden_lines = set()
            fold_start_map = {}
            
            for start, end in fold_ranges:
                count = end - start + 1
                fold_start_map[start] = count
                for i in range(start, end + 1):
                    hidden_lines.add(i)

            # 6. Reconstruct the file
            lines = code.splitlines()
            result = []
            
            i = 0
            while i < len(lines):
                line_num_display = i + 1 # 1-based for display
                
                if i in hidden_lines:
                    # Only print the placeholder once per fold
                    if i in fold_start_map:
                        count = fold_start_map[i]
                        
                        # INDENTATION FIX:
                        # Grab indentation from the PREVIOUS line (the function signature)
                        # so the "// ... folded" comment aligns nicely.
                        prev_indent = ""
                        if i > 0:
                            prev_line = lines[i-1]
                            prev_indent = prev_line[:len(prev_line) - len(prev_line.lstrip())]
                        
                        # Add a distinct marker for the LLM
                        result.append(f" ... | {prev_indent}// ... logic folded ({count} lines) ...")
                    
                    # Skip the actual content
                    i += 1
                else:
                    result.append(f"{line_num_display:4d} | {lines[i]}")
                    i += 1
            
            return "\n".join(result)

        except Exception as e:
            import traceback
            return f"Error generating skeleton: {e}\n{traceback.format_exc()}"

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
