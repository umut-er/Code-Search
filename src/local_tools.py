import os
import subprocess
import json
import difflib
import math
from typing import List, Dict, Any

from langchain_core.tools import tool

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser
from openai import OpenAI
import time

PROJECT_ROOT = "./bilkent-tanitim/frontend"

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def generate_skeleton(path: str, code: str) -> str:
    """
    Internal helper: Generates a skeleton view of the code by folding function bodies.
    Returns the skeleton string. If language is not supported, returns a specific error string.
    """
    ext = os.path.splitext(path)[1].lower()
    valid_exts = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
    
    if ext not in valid_exts:
        return f"Error: Skeleton view only supports JS/TS files. Got {ext}."

    lang_name = "typescript" if ext in [".ts", ".tsx"] else "javascript"

    try:
        parser = get_parser(lang_name)
        language = get_language(lang_name)
        tree = parser.parse(bytes(code, "utf8"))
        
        # Query for foldable blocks
        query_scm = """
        (function_declaration body: (_) @body)
        (method_definition body: (_) @body)
        (arrow_function body: (_) @body)
        (class_declaration body: (_) @body)
        """

        query = Query(language, query_scm)
        cursor = QueryCursor(query)
        captures_obj = cursor.captures(tree.root_node)
        
        nodes = []
        if isinstance(captures_obj, dict):
            for _, captured_nodes in captures_obj.items():
                nodes.extend(captured_nodes)
        else:
            nodes = [node for node, _ in captures_obj]

        # Fold large top-level variables (e.g. styled components)
        program = tree.root_node
        for child in program.children:
            if child.type in ("lexical_declaration", "variable_declaration"):
                if (child.end_point[0] - child.start_point[0]) > 4:
                    nodes.append(child)

        nodes.sort(key=lambda n: n.start_point[0])
        fold_ranges = []
        last_fold_end_line = -1

        for node in nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            if (end_line - start_line) < 5: continue
            if start_line <= last_fold_end_line: continue
            
            fold_start = start_line + 1
            fold_end = end_line - 1

            # Smart logic: Keep 'return' statement visible if found
            if node.type == "statement_block":
                return_node = None
                for child in node.children:
                    if child.type == "return_statement":
                        return_node = child
                
                if return_node:
                    candidate_end = return_node.start_point[0] - 1
                    if candidate_end > fold_start:
                        fold_end = candidate_end

            if fold_start < fold_end:
                fold_ranges.append((fold_start, fold_end))
                last_fold_end_line = end_line 

        # Reconstruct
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
        return f"Error generating skeleton: {str(e)}"

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
        target_path = path
        
        if not os.path.exists(target_path):
            potential_path = os.path.join(PROJECT_ROOT, path)
            if os.path.exists(potential_path):
                target_path = potential_path
                
        if not os.path.exists(target_path):
            return f"Error: File not found at {path} (checked {target_path})"
            
        path = target_path

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
            target_path = path

            if not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path

            if not os.path.exists(target_path):
                return f"Error: File not found at {path} (checked {target_path})"
            
            # Use target_path (the resolved path) instead of the original input path
            with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
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
            target_path = path

            if not os.path.isdir(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.isdir(potential_path):
                    target_path = potential_path

            if not os.path.isdir(target_path):
                return f"Error: {path} is not a directory (checked {target_path})."
            
            # Use target_path to list the actual contents
            items = os.listdir(target_path)
            formatted = []
            for item in items:
                # Skip hidden files like .git to save noise
                if item.startswith("."): 
                    continue
                    
                full_path = os.path.join(target_path, item)
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
            target_path = path

            if path == "." or not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            if not os.path.exists(target_path):
                target_path = PROJECT_ROOT

            results = []
            # Common web dev folders to ignore to speed up search
            exclude_dirs = {
                "node_modules", ".git", ".next", "dist", "build", 
                "coverage", ".vscode", "__pycache__"
            }
            
            # 1. First pass: exact substring matching (fastest/most accurate)
            # 2. Second pass: fuzzy matching if substring fails
            
            candidates = []
            
            for root, dirs, files in os.walk(target_path):
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
            target_path = path

            if path == "." or not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            if not os.path.exists(target_path):
                target_path = PROJECT_ROOT

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
                target_path,  # Use resolved path here
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
        try:
            target_path = path
            if not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path

            if not os.path.exists(target_path):
                return f"Error: File not found at {path}"

            with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            
            return generate_skeleton(target_path, code)

        except Exception as e:
            return f"Error reading file skeleton: {str(e)}"

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
            target_path = path

            if path == "." or not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            if not os.path.exists(target_path):
                target_path = PROJECT_ROOT

            # Check for ripgrep
            if subprocess.call(["which", "rg"], stdout=subprocess.DEVNULL) != 0:
                return "Error: 'rg' (ripgrep) is not installed on this system."

            command = [
                "rg", 
                "-n", # Line numbers
                f"-C{context_lines}", # Context
                "--json", # JSON output for reliable parsing
                "-e", query, 
                target_path # Use the resolved path
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



    @tool("find_best_route")
    def find_best_route(bug_info: Dict[str, Any]) -> str:
        """
        Decides the best starting point for reproducing a bug using Semantic Search + LLM Reasoning.
        
        1. Embeds the bug description and all route metadata (Description + Usecases).
        2. Finds the top 7 most relevant routes using Cosine Similarity.
        3. Asks the LLM to select the single best match from these top candidates using Chain-of-Thought.

        Args:
            bug_info: The bug information object.
            
        Returns:
            A JSON string containing 'reasoning', 'path', and 'component'.
        """
        # --- START: APPROACH 1 PATH RESOLUTION ---
        routes_filename = "routes.json"
        target_path = routes_filename
        
        # 1. Check current directory
        if not os.path.exists(target_path):
            # 2. Check PROJECT_ROOT
            potential_path = os.path.join(PROJECT_ROOT, routes_filename)
            if os.path.exists(potential_path):
                target_path = potential_path
                
        if not os.path.exists(target_path):
            return json.dumps({"path": "/", "component": None, "error": "routes.json not found"})
        # --- END: APPROACH 1 PATH RESOLUTION ---

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                routes = json.load(f)
        except Exception:
            return json.dumps({"path": "/", "component": None})

        if not routes:
            return json.dumps({"path": "/", "component": None})

        try:

            client = OpenAI()
        except ImportError:
            return json.dumps({"path": "/", "component": None, "error": "OpenAI library not available"})

        # --- PHASE 1: SEMANTIC SEARCH (RAG) ---
        # Prepare text chunks for embedding
        # We combine path, description, and usecases into a single semantic string for each route.
        # Handle null values safely
        def safe_get(obj, key, default=""):
            """Safely get value from dict, handling None/null values."""
            value = obj.get(key)
            return value if value is not None else default
        
        # Build bug description with null-safe handling
        bug_parts = []
        title = safe_get(bug_info, "Title")
        if title:
            bug_parts.append(f"Title: {title}")
        
        description = safe_get(bug_info, "Description")
        if description:
            bug_parts.append(f"Description: {description}")
        
        s2r = safe_get(bug_info, "S2R")
        if s2r:
            bug_parts.append(f"Steps to Reproduce: {s2r}")
        
        ob = safe_get(bug_info, "OB")
        if ob:
            bug_parts.append(f"Observed Behavior: {ob}")
        
        eb = safe_get(bug_info, "EB")
        if eb:
            bug_parts.append(f"Expected Behavior: {eb}")
        
        bug_description = "\n".join(bug_parts) if bug_parts else "No bug information available"
        
        route_texts = []
        for r in routes:
            desc = r.get('description', '')
            usecases = ", ".join(r.get('usecases', []))
            # Intent-focused text representation
            text_repr = f"Route Path: {r.get('path')}\nPurpose: {desc}\nCapabilities: {usecases}"
            route_texts.append(text_repr)

        try:
            # 1. Embed the Bug Description
            bug_emb_resp = client.embeddings.create(
                input=[bug_description],
                model="text-embedding-3-small" # Cheap and fast
            )
            bug_vector = bug_emb_resp.data[0].embedding

            # 2. Embed All Routes (Batch Processing)
            # Note: If you have >2000 routes, you might need to batch this loop. 
            # For <100 routes, a single call is fine.
            route_emb_resp = client.embeddings.create(
                input=route_texts,
                model="text-embedding-3-small"
            )
            route_vectors = [item.embedding for item in route_emb_resp.data]

            # 3. Calculate Similarity Scores
            scored_routes = []
            for i, r_vector in enumerate(route_vectors):
                score = cosine_similarity(bug_vector, r_vector)
                scored_routes.append((score, routes[i]))

            # 4. Get Top K Candidates (e.g., Top 7)
            # We assume the correct route is definitely within the top 7 semantic matches.
            scored_routes.sort(key=lambda x: x[0], reverse=True)
            top_routes = [item[1] for item in scored_routes[:7]]

        except Exception as e:
            print(f"Embedding/Search failed: {e}. Falling back to simple keyword search.")
            # Fallback: If embedding fails, just take the first 10 or do a basic keyword filter
            top_routes = routes[:10]

        # --- PHASE 2: LLM REASONING (Chain-of-Thought) ---
        
        # Format ONLY the top candidates for the prompt (Saving Tokens!)
        routes_context = []
        for r in top_routes:
            info = f"Path: {r.get('path')}\nComponent: {r.get('component')}\nDescription: {r.get('description')}\nUsecases:\n"
            for uc in r.get('usecases', []):
                info += f"  - {uc}\n"
            routes_context.append(info)
        
        context_text = "\n---\n".join(routes_context)

        prompt = f"""You are a Senior QA Automation Engineer.
    Your goal is to identify the **single best starting URL** to reproduce the bug described below.

    I have performed a semantic search and identified the following TOP {len(top_routes)} MOST RELEVANT ROUTES:

    {context_text}

    ---
    BUG REPORT:
    {bug_description}

    ---
    ### ANALYSIS STRATEGY (Chain of Thought):
    1. **Identify the Failure Location**: Where does the bug *manifest*?
    - If the bug says "After clicking submit on the login page, it crashes", the start route is likely `/login`.
    - If it says "The dashboard graph is empty", the start route is likely `/dashboard`.

    2. **Analyze Intent**: Compare the bug's intent with the 'Description' and 'Usecases' provided above.
    - Ignore generic features (like "User can view header") unless the bug is specifically about the header.
    - Focus on unique business logic (e.g., "High School Application form").

    3. **Handle Global Issues**: 
    - If the bug is about a global component (Sidebar, Navbar) appearing on all pages, prefer the root path "/" or "/dashboard".

    ### OUTPUT
    Return a JSON object with:
    - "reasoning": Brief explanation of why you selected this route over others.
    - "path": The URL path.
    - "component": The file path.

    Example:
    {{
    "reasoning": "The bug reports a failure when submitting the high school form. The '/highschool-application' route specifically lists 'Submit application' in its use cases, making it the most direct match.",
    "path": "/highschool-application",
    "component": "src/modules/tour-application/HighSchoolApplication.tsx"
    }}
    """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a route selection expert. Return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in find_best_route LLM call: {e}")
            # Fallback to the top semantic match if LLM fails
            best_match = top_routes[0]
            return json.dumps({
                "path": best_match.get("path"),
                "component": best_match.get("component"),
                "reasoning": "LLM failed, returning top semantic match."
            })

    @tool("submit_final_report")
    def submit_final_report(bug_info: Dict[str, Any], file_paths: List[str]) -> str:
        """
        The FINAL ACTION tool. Call this when you have identified all relevant file paths.
        
        This tool will:
        1. Read the full content of the files.
        2. Generate a SKELETON view of the code to save tokens.
        3. Use an LLM to generate a concise 'description' of why the file is relevant based on the skeleton.
        4. Save the full context to disk.

        Args:
            bug_info: The enhanced bug report object (must contain 'Title', 'OB', 'S2R', etc.).
            file_paths: A list of relative file paths relevant to the bug (e.g., ["src/components/Login.tsx"]).
        """
        # Initialize OpenAI
        try:
            client = OpenAI()
        except Exception as e:
            return f"Error initializing OpenAI: {e}"

        # Helper function for safe get with default
        def safe_get_bug(key, default=None):
            value = bug_info.get(key)
            return value if value is not None else default
        
        final_output = {
            "ID": safe_get_bug("ID"),
            "Title": safe_get_bug("Title"),
            "Description": safe_get_bug("Description"),
            "OB": safe_get_bug("OB"),
            "EB": safe_get_bug("EB"),
            "S2R": safe_get_bug("S2R"),
            "relevant_files": []
        }

        print("\n📝 Generating Final Report Descriptions...")

        for path in file_paths:
            time.sleep(21)
            content = ""
            target_path = path
            
            # Resolve Path
            if not os.path.exists(target_path):
                potential_path = os.path.join(PROJECT_ROOT, path)
                if os.path.exists(potential_path):
                    target_path = potential_path
            
            # Read Content
            if os.path.exists(target_path):
                try:
                    with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    content = f"<Error reading file: {str(e)}>"
            else:
                content = "<Error: File not found on disk>"
                final_output["relevant_files"].append({
                    "name": os.path.basename(path),
                    "path": path,
                    "description": "File not found.",
                    "content": content
                })
                continue

            # --- GENERATE SMART CONTEXT FOR LLM ---
            # Try to generate a skeleton. If it fails (e.g. CSS/JSON file), use truncated content.
            context_for_llm = generate_skeleton(target_path, content)
            
            if context_for_llm.startswith("Error:"):
                # Fallback for non-JS/TS files: use first 500 lines
                lines = content.splitlines()
                context_for_llm = "\n".join(lines[:500])
                if len(lines) > 500:
                    context_for_llm += "\n... (remaining content truncated for description generation) ..."

            # --- LLM CALL FOR DESCRIPTION ---
            # Build bug report section with null-safe handling
            bug_report_parts = []
            title = safe_get_bug('Title')
            if title:
                bug_report_parts.append(f"Title: {title}")
            
            desc = safe_get_bug('Description')
            if desc:
                bug_report_parts.append(f"Description: {desc}")
            
            s2r = safe_get_bug('S2R')
            if s2r:
                bug_report_parts.append(f"Steps to Reproduce: {s2r}")
            else:
                bug_report_parts.append("Steps to Reproduce: Not provided")
            
            ob = safe_get_bug('OB')
            if ob:
                bug_report_parts.append(f"Observed Behavior: {ob}")
            else:
                bug_report_parts.append("Observed Behavior: Not provided")
            
            bug_report_text = "\n".join(bug_report_parts) if bug_report_parts else "No bug information available"
            
            prompt = f"""
            You are a Technical QA Lead.
            
            BUG REPORT:
            {bug_report_text}
            
            FILE: {path}
            CODE SKELETON / CONTENT:
            {context_for_llm}
            
            TASK:
            Write a single, concise sentence explaining strictly WHY this file is relevant to this bug.
            Example: "Contains the 'LoginForm' component which is likely to used for the login process."
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=80
                )
                generated_description = response.choices[0].message.content.strip()
            except Exception as e:
                generated_description = f"Auto-generation failed: {str(e)}"

            # Append to Output
            final_output["relevant_files"].append({
                "name": os.path.basename(path),
                "path": path,
                "description": generated_description,
                "content": content # Save FULL content for the reproduction agent
            })

        # Save Final Context
        output_filename = "final_context.json"
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2)
            return "SEARCH_COMPLETED_SUCCESSFULLY"
        except Exception as e:
            return f"ERROR: Could not save context: {e}"