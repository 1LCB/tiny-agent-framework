from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDeltaToolCall
from typing import AsyncGenerator
from taf.constants import ConversationRoles, HookTypes
from taf.hooks import HookFunctions
from taf.skills import Skill
from taf.types import AgentChunkResponse
from taf.tools import ToolUtils
from openai import AsyncOpenAI
import json, asyncio


class Agent:
    """
    A small AI agent framework for building conversational agents with tool calling capabilities.
    """
    def __init__(self, name: str, model: str, system_prompt: str, tools: list = None, skills: list[Skill] = None, temperature: float = 0.2, **openai_kwargs):
        self.name = name
        self.model = model
        self.__tool_mapping = {}
        self.__tool_schemas = []
        self.temperature = temperature

        self.skills = skills or []
        if self.skills:
            self.__setup_skills(skills)

        self.hooks: dict[HookTypes, HookFunctions] = {}

        self.__llm_system_prompt = system_prompt
        self.__openai_client = AsyncOpenAI(**openai_kwargs)
        self.__conversation_history = [{"role": ConversationRoles.SYSTEM, "content": self.__llm_system_prompt}]

        tools = tools or []
        for tool in tools:
            self.__add_available_tool(tool)

    def __repr__(self):
        return f"<Agent name={self.name}, model={self.model}>"

    def __add_available_tool(self, function, strict: bool = False):
        self.__tool_mapping[function.__name__] = {
            "func": function,
            "hasContext": ToolUtils.has_ctx_parameter(function),
        }

        tool_schema = ToolUtils.function_to_openai_schema(function, strict=strict)
        self.__tool_schemas.append(tool_schema)

    def __setup_skills(self, skills: list[Skill]):
        if not skills:
            return

        def skill(skill_name: str, resource_path: str = None):
            """
            Load a skill or one of its resources.

            :param skill_name: Name of the skill
            :param resource_path: Optional relative path of a resource
            """
            import os

            existing_skill = next((s for s in skills if s.name == skill_name), None)
            if not existing_skill:
                return (
                    f'Skill "{skill_name}" not found. '
                    f"Available skills: {[s.name for s in skills]}."
                )

            skill_dir = os.path.dirname(existing_skill.file_path)

            if resource_path:
                if resource_path not in existing_skill.resources:
                    return (
                        f'Resource "{resource_path}" not found for skill "{skill_name}". '
                        f"Available resources: {existing_skill.resources}."
                    )

                full_path = os.path.join(skill_dir, resource_path)
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()

            content = existing_skill.load()
            if existing_skill.resources:
                content += "\n\n---\n\n## Available Resources (relative path)\n"
                for r in existing_skill.resources:
                    content += f"- {r}\n"
            return content

        self.__add_available_tool(skill)

    def tool(self, *, strict: bool = False):
        def wrapper(f):
            self.__add_available_tool(f, strict)
            return f
        return wrapper

    def hook(self, hook_type: HookTypes):
        def wrapper(f):
            self.hooks[hook_type] = HookFunctions(f)
            return f
        return wrapper
        
    def system_prompt(self):
        def wrapper(f):
            self.hooks[HookTypes.SYSTEM_PROMPT] = HookFunctions(f)
            return f
        return wrapper


    async def run_stream(
        self, 
        prompt: str, 
        dependency=None, 
        max_steps: int = 30,
        **kwargs
    ) -> AsyncGenerator[AgentChunkResponse, None]:
        system_prompt = (await self.__format_system_prompt(dependency, self.__llm_system_prompt)).strip()
        if system_prompt:
            self.__conversation_history[0]["content"] = system_prompt
        self.__append_message(ConversationRoles.USER, prompt)

        for step in range(max_steps):
            response_message = ""
            tool_calls_aggregator: list[ChoiceDeltaToolCall] = []

            stream = await self.__generate_stream_completion(conversation=self.__conversation_history, **kwargs)
            async for chunk in stream:
                if len(chunk.choices) == 0:
                    continue

                stream_chunk = chunk.choices[0]

                # some reasoning models
                if getattr(stream_chunk.delta, "reasoning", False):
                    reasoning = stream_chunk.delta.reasoning
                    if reasoning:
                        yield AgentChunkResponse(origin=self.name, content=reasoning, metadata={}, type="reasoning")

                content_delta = stream_chunk.delta.content
                if content_delta:
                    response_message += content_delta
                    yield AgentChunkResponse(origin=self.name, content=content_delta, metadata={}, type="response")

                tool_call_chunks = stream_chunk.delta.tool_calls
                if tool_call_chunks:
                    tool_calls_aggregator.extend(tool_call_chunks)

            self.__append_message(ConversationRoles.ASSISTANT, response_message, tool_calls_aggregator)
            if not tool_calls_aggregator:
                break # final answer

            async for tool_calls in self.__handle_tool_calls(dependency, tool_calls_aggregator):
                yield tool_calls

    async def __handle_tool_calls(self, dependency, tool_calls_aggregator: list[ChoiceDeltaToolCall]):
        for tool_call in tool_calls_aggregator:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = tool_call.id

            meta = {"tool_name": tool_name, "tool_args": tool_args, "tool_id": tool_id}
            yield AgentChunkResponse(origin=self.name, content=None, metadata=meta, type="tool_call")

            output = await self.__execute_func(tool_name, tool_args, dependency)

            yield AgentChunkResponse(origin=self.name, content=output, metadata=meta, type="tool_call_result")

            self.__conversation_history.append({
                "role": ConversationRoles.TOOL,
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(output),
            })

    async def __call_hook(self, hook_type: HookTypes, **params):
        hook = self.hooks.get(hook_type)
        if not hook:
            return None
        return await hook.call(**params)

    async def __format_system_prompt(self, dependency, base_prompt: str):
        output = await self.__call_hook(HookTypes.SYSTEM_PROMPT, ctx=dependency)
        if not output: 
            return base_prompt

        base_prompt += "\n" + output
        return base_prompt

    async def __execute_func(self, tool_name: str, tool_parameters_str: str, dependency=None):
        func_data = self.__tool_mapping.get(tool_name)
        if not func_data:
            return f"Error: Tool '{tool_name}' not found."

        func = func_data["func"]
        has_context = func_data["hasContext"]
        try:
            params = json.loads(tool_parameters_str)
            if asyncio.iscoroutinefunction(func):
                result = await func(**params, ctx=dependency) if has_context else await func(**params)
            else:
                result = func(**params, ctx=dependency) if has_context else func(**params)
            return result
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments provided."
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    async def __generate_stream_completion(self, conversation: list, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        completion = await self.__openai_client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
            tools=self.__tool_schemas if self.__tool_schemas else None,
            tool_choice="auto" if self.__tool_schemas else None,
            n=1,
            **kwargs,
        )
        return completion

    def __append_message(self, role: str, content: str | None, tool_calls=None) -> None:
        message = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
            message["content"] = None
        self.__conversation_history.append(message)

    def clear_history(self) -> None:
        self.__conversation_history = [{"role": ConversationRoles.SYSTEM, "content": self.__llm_system_prompt}]

    def export_conv(self, include_system_prompt: bool = False, include_tool_calls: bool = True) -> list[dict]:
        conv = self.__conversation_history.copy()
        if not include_tool_calls:
            conv = [msg for msg in conv if msg["role"] != ConversationRoles.TOOL]
        if not include_system_prompt:
            del conv[0]
        return conv

    def load_conv(self, conv: list):
        self.clear_history()
        self.__conversation_history.extend(conv)
