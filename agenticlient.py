# agentic_client.py
import asyncio
import boto3
from fastmcp import FastMCP, Client
from typing import Dict, Any, List

# --- SERVER DEFINITION ---
mcp_server = FastMCP("Demo Calculator")

@mcp_server.tool
def jama(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: The first number.
        b: The second number.
    """
    print("hereh")
    return a + b

@mcp_server.tool
def minha(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: The first number.
        b: The second number.
    """
    print("hereh 111")
    return a - b

# --- AGENTIC CLIENT LOGIC (using AWS Bedrock) ---
async def main():
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    async with Client(mcp_server) as client:
        print("Agent connected to the in-memory MCP server.")

        tools = await client.list_tools()
        
        tool_definitions: List[Dict[str, Any]] = []
        for tool in tools:
            tool_schema = tool.model_dump(by_alias=True)
            function_spec = tool_schema.get('function', {}) 
            parameters_schema = function_spec.get('parameters', {})

            # CRITICAL FIX 1: Ensure root has 'type': 'object'
            if "type" not in parameters_schema:
                parameters_schema["type"] = "object"
            
            # Map the components and ADD the required 'inputSchema' structure
            bedrock_spec = {
                "name": function_spec.get('name') or tool.name, 
                "description": function_spec.get('description') or tool.description,
                "inputSchema": {
                    "json": parameters_schema # Pass the non-null schema here
                }
            }
            
            # Debugging print statement: inspect the generated schema before AWS call
            print(f"Generated Bedrock Spec for '{bedrock_spec['name']}':\n{json.dumps(bedrock_spec, indent=2)}")

            tool_definitions.append({
                "toolSpec": bedrock_spec
            })
        
        user_query = "100 and 50 are my favoirte number?"
        messages = [
            {
                "role": "user", 
                "content": [
                    {"text": "You are a calculator assistant. Use the tools provided for ALL math. Do not calculate manually."}
                ]
            },
            {"role": "user", "content": [{"text": user_query}]}
        ]

        print(f"\nUser query: '{user_query}'")

        # The Agentic Loop (same as before)
        while True:
           

            response = bedrock_runtime.converse(
                modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                messages=messages,
                toolConfig={"tools": tool_definitions},
            )

            # ... rest of the agentic loop remains the same ...

            print(response)
            response_message = response['output']['message']
            messages.append(response_message)
            response_content_blocks = response_message['content']

            if any(item.get('text') for item in response_content_blocks):
                print("parsing the structure")
                final_answer = next(item['text'] for item in response_content_blocks if item.get('text'))
                print(f"\n✅ Final Answer from LLM: {final_answer}")
                #break
            if response['stopReason'] == "end_turn":
                final_answer = next((item['text'] for item in response_message['content'] if 'text' in item), "No text found")
                print(f"\n✅ Final Answer: {final_answer}")
                break
            
            # Process tool calls
            if response['stopReason'] == "tool_use":
                tool_results_content = []
                for content_block in response_message['content']:
                    if 'toolUse' in content_block:
                        tool_use = content_block['toolUse']
                        
                        # Fix 4: Check if input is empty and handle it (or debug)
                        if not tool_use.get('input'):
                            print("⚠️ Warning: LLM requested tool but sent empty input parameters.")
                            continue

                        print(f"🛠️ Executing: {tool_use['name']} with {tool_use['input']}")
                        result_value = await client.call_tool(tool_use['name'], tool_use['input'])
                        
                        tool_results_content.append({
                            "toolResult": {
                                "toolUseId": tool_use['toolUseId'],
                                "content": [{"text": str(result_value)}]
                            }
                        })
                
                # Append result to messages and continue loop
                if tool_results_content:
                    messages.append({"role": "user", "content": tool_results_content})
                    print(messages)
                    break


            
            
    print("\nAgent process complete.")

if __name__ == "__main__":
    import json # Import json at the top if needed for the debug print
    asyncio.run(main())
