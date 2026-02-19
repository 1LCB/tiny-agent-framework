from taf.agent import Agent

class AgentChain:
    def __init__(self, dependency = None):
        self.dependency = dependency
        self.agents: list[tuple[Agent, str]] = []

    async def __get_agent_final_answer(self, agent: Agent, prompt, dependency):
        agent_response = ""
        async for chunk in agent.run_stream(prompt, dependency):
            if chunk["type"] == "response":
                agent_response += chunk["content"]
        return agent_response

    def next(self, agent: Agent, task: str):
        previous_agent = self.agents[-1][0] if self.agents else None

        if previous_agent:
            @agent.tool()
            async def ask_previous_agent(question: str):
                """
                Ask the previous agent a specific question about its work.

                Use this if the provided previous result is not enough and you
                need clarification or additional details. Returns the previous
                agent's final answer to your question.
                """
                response = await self.__get_agent_final_answer(previous_agent, question, self.dependency)
                return {"agent_response": response}

        self.agents.append((agent, task))
        return self

    async def result(self):
        if not self.agents:
            raise Exception(f"No agents set for this chain.")

        previous_result = None
    
        for agent, task in self.agents:
            if previous_result:
                task = f"""
<previous-result>
This is the result of the previous agent's work. You must use this as source of information to keep on your current task. In case you need more information from the previous agent, you can call the `ask_previous_agent` tool to make questions.

Previous Result:
{previous_result}
</previous-result>

Current Task:
{task}
""".strip()

            agent_response = ""
            async for chunk in agent.run_stream(task, dependency=self.dependency):
                if chunk["type"] in ("reasoning", "response"):
                    print(chunk["content"], end="", flush=True)
                    agent_response += chunk["content"]

                elif chunk["type"] in ("tool_call", "tool_call_result"):
                    print("\n", chunk["content"])

            previous_result = agent_response or "Task completed successfully. No response needed."
        return previous_result
