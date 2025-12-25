import inspect
from typing import get_args, get_origin, Literal, Union

class ToolUtils:
    @classmethod
    def function_to_openai_schema(cls, function):
        signature = inspect.signature(function)
        parameters = {}
        required = []

        for name, param in signature.parameters.items():
            if name == "ctx":
                continue # skip reserved context parameter in schema

            param_schema = cls._get_type_schema(param.annotation)
            parameters[name] = param_schema
            
            if param.default is inspect.Parameter.empty:
                required.append(name)

        tool_schema = {
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": inspect.getdoc(function),
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }
        return tool_schema

    @classmethod
    def _get_type_schema(cls, annotation):
        # handle None/empty annotation
        if annotation is inspect.Parameter.empty or annotation is None:
            return {"type": "string"}
        
        # handle Union types (including Optional)
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            # Optional[T] is Union[T, None]
            if len(args) == 2 and type(None) in args:
                non_none_type = next(arg for arg in args if arg is not type(None))
                return cls._get_type_schema(non_none_type)
            # For other unions, use the first type
            return cls._get_type_schema(args[0])
        
        # handle Literal enums
        if get_origin(annotation) is Literal:
            return {"type": "string", "enum": list(get_args(annotation))}
        
        # handle lists
        if get_origin(annotation) is list:
            args = get_args(annotation)
            item_schema = cls._get_type_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}
        
        # handle dicts
        if get_origin(annotation) is dict:
            args = get_args(annotation)
            value_schema = cls._get_type_schema(args[1]) if len(args) > 1 else {"type": "string"}
            return {"type": "object", "additionalProperties": value_schema}
        
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean"
        }
        return {"type": type_map.get(annotation, "string")}

    @classmethod
    def has_ctx_parameter(cls, func) -> bool:
        signature = inspect.signature(func)
        return "ctx" in signature.parameters
