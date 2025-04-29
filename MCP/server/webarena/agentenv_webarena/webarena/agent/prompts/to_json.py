import glob
import importlib
import json
import os
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# use the current directory as the root
def run() -> None:
    """Convert all python files in agent/prompts to json files in agent/prompts/jsons

    Python files are easiser to edit
    """
    for p_file in glob.glob(f"/root/wrp/MCP-A2A/MCP/server/webarena/agentenv_webarena/webarena/agent/prompts/raw/*.py"):
        # import the file as a module
        print('p_file:',p_file)
        base_name = os.path.basename(p_file).replace(".py", "")
        module = importlib.import_module(f"agent.prompts.raw.{base_name}")
        prompt = module.prompt
        # save the prompt as a json file
        os.makedirs("/root/wrp/MCP-A2A/MCP/server/webarena/agentenv_webarena/webarena/agent/prompts/jsons", exist_ok=True)
        with open(f"/root/wrp/MCP-A2A/MCP/server/webarena/agentenv_webarena/webarena/agent/prompts/jsons/{base_name}.json", "w+") as f:
            json.dump(prompt, f, indent=2)
    print(f"Done convert python files to json")


if __name__ == "__main__":
    run()
