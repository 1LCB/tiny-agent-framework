import asyncio, inspect

class HookFunctions:
    def __init__(self, func):
        self.func = func

    def __get_parameters(self):
        return inspect.signature(self.func).parameters
    
    async def call(self, **params):
        filtered_params = {
            k: v for k, v in params.items() 
            if k in self.__get_parameters()
        }
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**filtered_params)
        return self.func(**filtered_params)
