import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()  # load from .env if present

# Read Jira config from environment
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "BUG")  # default to BUG if not set

def _ensure_required_config():
    missing = []
    if not JIRA_SERVER:
        missing.append("JIRA_SERVER")
    if not JIRA_EMAIL:
        missing.append("JIRA_EMAIL")
    if not JIRA_API_TOKEN:
        missing.append("JIRA_API_TOKEN")
    if missing:
        raise RuntimeError(
            f"Missing required Jira configuration in environment: {', '.join(missing)}. "
            "Create a .env file with these keys or set them in your environment."
        )

def adf_to_text(adf):
    """
    Convert Atlassian Document Format (ADF) JSON to plain text.
    Handles paragraphs and basic text nodes.
    """

    # If Jira is already giving a plain string, just return it
    if isinstance(adf, str):
        return adf or ""

    if not isinstance(adf, dict):
        return ""

    lines = []

    def walk(node, in_paragraph=False):
        nonlocal lines

        if isinstance(node, dict):
            node_type = node.get("type")

            # Paragraph: collect its children into one line
            if node_type == "paragraph":
                buf = []
                for child in node.get("content", []):
                    text = walk(child, in_paragraph=True)
                    if text:
                        buf.append(text)
                if buf:
                    lines.append("".join(buf))
                return ""

            # Text node: return its text
            if node_type == "text":
                return node.get("text", "")

            # Other nodes: just walk their content
            out = []
            for child in node.get("content", []):
                t = walk(child, in_paragraph=in_paragraph)
                if t:
                    out.append(t)
            return "".join(out)

        elif isinstance(node, list):
            for child in node:
                walk(child, in_paragraph=in_paragraph)
        return ""

    walk(adf)
    return "\n".join(lines)


def fetch_bug_reports():
    _ensure_required_config()
    url = f"{JIRA_SERVER}/rest/api/3/search/jql"

    jql = f"project = {PROJECT_KEY} AND issuetype = Bug ORDER BY created DESC"

    params = {
        "jql": jql,
        "maxResults": 50,
        "fields": "summary,description,status,reporter,created",
    }

    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers, params=params, auth=auth)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Response body:")
        print(response.text)
        return []

    data = response.json()
    issues = data.get("issues", [])
    results = []

    for issue in issues:
        fields = issue["fields"]
        raw_desc = fields.get("description")

        # Convert ADF JSON to plain text
        description_text = adf_to_text(raw_desc)

        results.append({
            "key": issue["key"],
            "summary": fields.get("summary"),
            "description": description_text,
            "status": fields.get("status", {}).get("name"),
            "reporter": fields.get("reporter", {}).get("displayName"),
            "created": fields.get("created"),
        })

    return results

def _get_transitions(issue_key: str):
    """
    Retrieve available transitions for a Jira issue.
    """
    _ensure_required_config()
    url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_key}/transitions"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}
    resp = requests.get(url, headers=headers, auth=auth)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("transitions", [])

def transition_issue_to(issue_key: str, target_name: str = "In Progress") -> bool:
    """
    Transition the Jira issue to the given status by matching transition name.
    """
    transitions = _get_transitions(issue_key)
    target = None
    for t in transitions:
        name = (t.get("name") or "").strip()
        if name.lower() == target_name.lower():
            target = t
            break
    if not target:
        return False
    transition_id = target.get("id")
    if not transition_id:
        return False
    url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_key}/transitions"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"transition": {"id": transition_id}}
    resp = requests.post(url, headers=headers, auth=auth, json=payload)
    return 200 <= resp.status_code < 300

def _text_to_adf(text: str) -> dict:
    """
    Wrap plain text into minimal ADF document for Jira comments.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": text or ""}
                ]
            }
        ]
    }

def add_comment_to_issue(issue_key: str, comment_text: str) -> bool:
    """
    Add a comment to a Jira issue (Cloud) using ADF body.
    Returns True on success, False otherwise.
    """
    _ensure_required_config()
    url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_key}/comment"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {"body": _text_to_adf(comment_text)}
    try:
        resp = requests.post(url, headers=headers, auth=auth, json=payload)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


if __name__ == "__main__":
    bugs = fetch_bug_reports()

    if not bugs:
        print("No bug reports found (or the query returned empty).")
    else:
        print(f"Fetched {len(bugs)} bug(s):")
        for bug in bugs:
            print("------------------------------------")
            print("KEY:", bug["key"])
            print("Summary:", bug["summary"])
            print("Status:", bug["status"])
            print("Reporter:", bug["reporter"])
            print("Created:", bug["created"])
            print("Description:")
            print(bug["description"])