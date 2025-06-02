import os

BASE_DIR = "scholar_verse"

# Define the agent and sub-agent structure
AGENTS = {
    "router_agent": {
        "files": ["agent.py", "prompt.py", "tools.py"],
        "sub_agents": {
            "ingestion_agent": ["agent.py", "prompt.py", "tools.py"],
            "citation_graph_agent": ["agent.py", "prompt.py", "tools.py"],
            "cross_paper_analysis_agent": ["agent.py", "prompt.py", "tools.py"],
            "deep_search_agent": ["agent.py", "prompt.py", "tools.py"],
            "insight_agent": ["agent.py", "prompt.py", "tools.py"],
            "visualization_agent": ["agent.py", "prompt.py", "tools.py"]
        }
    }
}

# Other root-level directories
ROOT_FOLDERS = [
    "config",
    "database",
    "tests",
    "deployment",
    "eval",
    "web_research_cache"
]

def create_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("")

def create_project_structure():
    print(f"Creating ScholarVerse structure under '{BASE_DIR}'...")
    os.makedirs(BASE_DIR, exist_ok=True)

    # Create agents and sub-agents
    agents_dir = os.path.join(BASE_DIR, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    open(os.path.join(agents_dir, "__init__.py"), "w").close()

    for agent_name, agent_data in AGENTS.items():
        agent_path = os.path.join(agents_dir, agent_name)
        os.makedirs(agent_path, exist_ok=True)

        # Main agent files
        for file in agent_data["files"]:
            create_file(os.path.join(agent_path, file))

        # Sub-agents
        sub_agents_path = os.path.join(agent_path, "sub_agents")
        os.makedirs(sub_agents_path, exist_ok=True)

        for sub_agent_name, files in agent_data["sub_agents"].items():
            sub_agent_path = os.path.join(sub_agents_path, sub_agent_name)
            os.makedirs(sub_agent_path, exist_ok=True)

            for file in files:
                create_file(os.path.join(sub_agent_path, file))

    # Create other root-level folders
    for folder in ROOT_FOLDERS:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    print("✅ ScholarVerse structure created successfully.")

if __name__ == "__main__":
    create_project_structure()
