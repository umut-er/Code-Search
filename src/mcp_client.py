import json
from typing import Any, List, Dict, Optional
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field
from mcp import ClientSession

class SimpleMCPClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def mcp_tool_wrapper(self, name: str, **kwargs) -> str:
        # print(f"\n🔵 [MCP CALL] {name}") # Optional: Comment out to reduce noise in Benchmark
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        try:
            result = await self.session.call_tool(name, arguments=clean_kwargs)
            
            final_text = ""
            if result.content:
                text_content = []
                for block in result.content:
                    if block.type == "text":
                        text_content.append(block.text)
                final_text = "\n".join(text_content)
            else:
                final_text = "<No Text Content Returned>"

            return final_text
            
        except Exception as e:
            return f"Error: {e}"

    def convert_to_langchain_tool(self, mcp_tool_def: Any) -> StructuredTool:
        name = mcp_tool_def.name
        description = mcp_tool_def.description or f"MCP Tool: {name}"
        schema = mcp_tool_def.inputSchema
        
        required_fields = schema.get("required", [])
        fields = {}
        
        if "properties" in schema:
            for field_name, field_info in schema["properties"].items():
                t = field_info.get("type")
                if t == "integer": field_type = int
                elif t == "number": field_type = float
                elif t == "boolean": field_type = bool
                elif t == "array": field_type = List[Any]
                elif t == "object": field_type = Dict[str, Any]
                else: field_type = str
                
                if field_name in required_fields:
                    fields[field_name] = (field_type, Field(..., description=field_info.get("description", "")))
                else:
                    fields[field_name] = (Optional[field_type], Field(default=None, description=field_info.get("description", "")))
        
        ArgsModel = create_model(f"{name}Args", **fields)

        async def tool_func(**kwargs):
            return await self.mcp_tool_wrapper(name, **kwargs)

        return StructuredTool.from_function(
            func=None,
            coroutine=tool_func,
            name=name,
            description=description,
            args_schema=ArgsModel
        )
