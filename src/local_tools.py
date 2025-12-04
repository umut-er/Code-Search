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
        Reads a JS/TS file and returns a skeleton view.
        It folds logic bodies (hooks, handlers) but keeps the 'return (...)' JSX visible.
        
        Args:
            path: Relative path to the file.
        """
        if not os.path.exists(path):
            return f"Error: File not found at {path}"

        # 1. Strict JS/TS Filter
        ext = os.path.splitext(path)[1].lower()
        valid_exts = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
        
        if ext not in valid_exts:
            return f"Error: Skeleton view only supports JS/TS files. Use read_file for {ext}."

        lang_name = "typescript" if ext in [".ts", ".tsx"] else "javascript"

        try:
            parser = get_parser(lang_name)
            language = get_language(lang_name)
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            
            tree = parser.parse(bytes(code, "utf8"))
            
            # 2. Query for Function/Class Bodies
            query_scm = """
            (function_declaration body: (_) @body)
            (method_definition body: (_) @body)
            (arrow_function body: (_) @body)
            (class_declaration body: (_) @body)
            """

            # 3. Execute Query (Fixed for tree-sitter v0.22+)
            query = Query(language, query_scm)
            cursor = QueryCursor(query)
            captures_obj = cursor.captures(tree.root_node)
            
            # Normalize captures
            nodes = []
            if isinstance(captures_obj, dict):
                for _, captured_nodes in captures_obj.items():
                    nodes.extend(captured_nodes)
            else:
                nodes = [node for node, _ in captures_obj]

            # 4. Optional: Fold large top-level variables (Styled Components, Configs)
            program = tree.root_node
            for child in program.children:
                if child.type in ("lexical_declaration", "variable_declaration"):
                    if (child.end_point[0] - child.start_point[0]) > 4:
                        nodes.append(child)

            # 5. Process Folds
            nodes.sort(key=lambda n: n.start_point[0])
            fold_ranges = []
            last_fold_end_line = -1

            for node in nodes:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                
                # Skip tiny blocks (< 5 lines) or nested blocks
                if (end_line - start_line) < 5: continue
                if start_line <= last_fold_end_line: continue
                
                fold_start = start_line + 1
                fold_end = end_line - 1

                # --- SMART LOGIC: FIND RETURN STATEMENT ---
                # If we are inside a function body, look for the JSX return.
                if node.type == "statement_block":
                    return_node = None
                    # Search children for the LAST return statement
                    for child in node.children:
                        if child.type == "return_statement":
                            return_node = child
                    
                    if return_node:
                        # Fold everything up to the line BEFORE the return
                        candidate_end = return_node.start_point[0] - 1
                        if candidate_end > fold_start:
                            fold_end = candidate_end

                if fold_start < fold_end:
                    fold_ranges.append((fold_start, fold_end))
                    last_fold_end_line = end_line 

            # 6. Reconstruct File
            # Map: line_index -> lines_hidden_count
            fold_map = {start: (end - start + 1) for start, end in fold_ranges}
            hidden_lines = set()
            for start, end in fold_ranges:
                for i in range(start, end + 1):
                    hidden_lines.add(i)

            lines = code.splitlines()
            result = []
            i = 0
            
            while i < len(lines):
                if i in hidden_lines:
                    if i in fold_map:
                        count = fold_map[i]
                        # Calculate indentation from previous line
                        prev_indent = ""
                        if i > 0:
                            prev = lines[i-1]
                            prev_indent = prev[:len(prev) - len(prev.lstrip())]
                        
                        result.append(f" ... | {prev_indent}// ... logic folded ({count} lines) ...")
                    i += 1
                else:
                    result.append(f"{i+1:4d} | {lines[i]}")
                    i += 1
            
            return "\n".join(result)

        except Exception as e:
            import traceback
            return f"Error generating skeleton: {e}\n{traceback.format_exc()}"

    @tool("grep_text")
    def grep_text(query: str, path: str = ".", context_lines: int = 1) -> str:
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
