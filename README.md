# Tiny Agent Framework (TAF)

## Overview
A lightweight AI agent framework featuring tool calling, streaming responses, hooks and skills. Designed for simplicity and flexibility.

The framework operates through iterative (reAct) problem-solving by calling available tools. When no further tools are needed, it returns the final response.

## Features
- Dynamic system prompt composition
- Automatic tool schema generation via decorators
- Minimal dependencies (OpenAI library only)
- Dependency injection through tools and hooks
- Conversation history management
- Streaming response support
- [Skills](https://agentskills.io/)

## Quick Example

```python
import asyncio
from datetime import datetime
from taf.agent import Agent

SYSTEM_PROMPT = """
You are a helpful AI assistant.
"""

async def read_file(file_path: str) -> str:
    """
    Reads and returns the contents of a file at the given file path.
    """
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as exc:
        return f"Error reading file: {exc}"


agent = Agent(
    model="gemini-2.5-flash",
    system_prompt=SYSTEM_PROMPT,
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    tools=[read_file],  # Pre-register read_file as a tool
)


@agent.tool()
async def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file at the specified file path.
    """
    try:
        with open(file_path, "w") as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as exc:
        return f"Error writing to file: {exc}"


@agent.system_prompt()
async def get_current_datetime():
    """
    Injects the current datetime into the system prompt.
    """
    now = datetime.now()
    return f"Current datetime: {now.isoformat()}"


async def main():
    prompt = "Write a Python function that outputs 'hello world' inside a file named script.py"

    print("Agent response:")
    async for chunk in agent.run_stream(prompt):
        if chunk["type"] in ("reasoning", "response"):
            print(chunk["content"], end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Example using Dependency Injection
```python
import asyncio
from dataclasses import dataclass
from taf.agent import Agent

@dataclass
class UserData:
    id: int
    name: str
    language: str
    session_id: str

agent = Agent(
    model="gpt-4",
    system_prompt="You are a helpful AI assistant that personalizes responses based on user context.",
)


@agent.system_prompt()
async def get_current_user_information(ctx: UserData) -> str:
    """
    Injects current user information into the system prompt.
    """
    return f"""
Current User Context:
- User ID: {ctx.id}
- Name: {ctx.name}
- Preferred Language: {ctx.language}
- Session ID: {ctx.session_id}

Please tailor your responses to this user's context and speak in their preferred language when appropriate.
""".strip()


@agent.tool()
async def get_user_orders(ctx: UserData, status_filter: str = "all") -> str:
    """
    Retrieves user's orders filtered by status.
    """
    # In a real application, this would query a database
    return (
        f"Retrieved {status_filter} orders for user {ctx.name} "
        f"(ID: {ctx.id}): [Order1, Order2, Order3]"
    )


@agent.tool()
async def get_user_preferences(ctx: UserData) -> str:
    """
    Gets the current user's preferences and settings.
    """
    return f"""
User Preferences for {ctx.name}:
- Language: {ctx.language}
- Theme: Dark mode
- Notifications: Enabled
- Timezone: America/Sao_Paulo
""".strip()


async def main():
    user_context = UserData(
        id=1,
        name="Lucas",
        language="Brazilian Portuguese",
        session_id="sess_abc123xyz",
    )

    prompt = "What do you know about me and can you show my recent orders?"
    print("Agent: ", end="")

    async for chunk in agent.run_stream(prompt, dependency=user_context):
        if chunk["type"] in ("reasoning", "response"):
            print(chunk["content"], end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
```

## Example using Hooks
Hooks allow you to intercept and respond to various lifecycle events during agent execution. All hook functions must include the `metadata` parameter to receive event-specific data, and can optionally include `ctx` to access the dependency injection context.

Available hook types:
- `ON_USER_PROMPT` - Triggered when user submits a prompt
- `ON_AGENT_STEP` - Called at each reasoning step
- `ON_TOOL_CALL` - Before tool execution
- `ON_TOOL_CALL_RESULT` - After tool execution
- `ON_AGENT_FINAL_RESPONSE` - When agent provides final answer
- `SYSTEM_PROMPT` - Augments the system prompt dynamically

```python
import asyncio
from taf.agent import Agent
from taf.constants import HookTypes

agent = Agent(
    name="Agent",
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
)


@agent.hook(HookTypes.ON_TOOL_CALL)
def on_tool_call(metadata: dict):
    """Called before any tool is executed."""
    print(f"About to call tool: {metadata['tool_name']}")
    print(f"With arguments: {metadata['tool_args']}")


@agent.hook(HookTypes.ON_TOOL_CALL_RESULT)
def on_tool_result(metadata: dict):
    """Called after tool execution completes."""
    print(f"Tool {metadata['tool_name']} returned: {metadata['result']}")


@agent.hook(HookTypes.ON_AGENT_FINAL_RESPONSE)
def on_final_response(metadata: dict):
    """Called when agent provides its final answer."""
    print(f"Final response at step {metadata['step']}: {metadata['final_response']}")


@agent.hook(HookTypes.ON_USER_PROMPT)
def on_user_prompt(metadata: dict, ctx=None):
    """
    Called when user submits a prompt.
    Can access both metadata and dependency injection context.
    """
    print(f"User prompt: {metadata['prompt']}")
    if ctx:
        print(f"Context: {ctx}")
```

**Note:** Hook functions automatically receive only the parameters they declare. If your hook needs event data, include `metadata: dict`. If it needs dependency injection, include `ctx`. Both parameters are optional and will be filtered based on your function signature.

## Example using [Skills](https://agentskills.io/) (new)
Skills are installable units of knowledge that package task-specific instructions (SKILL.md) with optional supporting resources, loaded on demand to guide agent behavior while remaining **token-efficient**

Directory structure:
```
.skills/
├─ code-review/
│  └─ SKILL.md
└─ document-processing/
   ├─ references/
   │  ├─ CSV.md
   │  ├─ DOCX.md
   │  ├─ PDF.md
   │  └─ XLSX.md
   ├─ scripts/
   │  ├─ convert_document_to_image.sh
   │  └─ remove_images.py
   └─ SKILL.md
```

Code example:
```py
from taf import Agent, Skill
import asyncio

skills = Skill.from_folder(".skills")
print(f"{len(skills)} Skills loaded!")

agent = Agent(
    name="Skilled Agent",
    system_prompt=SYSTEM_PROMPT,
    model="",
    skills=skills
)

@agent.system_prompt()
def list_skills():
    content = "\n\n".join([f"Name: {i.name}\nDescription: {i.description}" for i in skills])
    return f"""
<available_skills>
{content}
</available_skills>
""".strip()

async def main():
    while True:
        prompt = input(">> ")
        async for chunk in agent.run_stream(prompt):
            if chunk["type"] in ("reasoning", "response"):
                print(i["content"], end="", flush=True)
        print()

asyncio.run(main())
```

### ⚠️ How Skills work under the hood:
- When skills are provided, a tool named `skill` is automatically added to the agent's toolset.
- Skills must be declared in the system prompt, so the model can discover them. If a skill is not listed the agent will not be aware of its existence.
- Skills are optional and replaceable. You can implement equivalent behaviour using tools, prompts or dynamic system prompt hooks without using the `Skill` class.
- For best results, include clear instructions in the system prompt that explain when and how the model should load and use skills.

## Disclaimer ⚠️
This project is primarily a proof-of-concept for building AI agents from scratch using Python and the OpenAI-compatible API. While functional, it may not be suitable for production environments without modifications.

Key limitations include:
- Basic error handling
- Limited memory management capabilities
- Minimal logging infrastructure
- Static tool registration


You are encouraged to adapt and extend the code according to your needs. The framework has been tested primarily with Gemini Flash and OpenRouter models; other OpenAI-compatible providers may require code adjustments.
