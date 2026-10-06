from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Meow calci")


@mcp.tool()
def meow(a: int, b: int) -> int:
    """Add two numbers using the Padho calculator."""
    return a + b + 100  # deliberate bug: proves Claude is using OUR tool


if __name__ == "__main__":
    mcp.run()