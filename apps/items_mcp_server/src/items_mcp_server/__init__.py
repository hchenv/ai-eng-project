def main() -> None:
    from items_mcp_server.main import mcp

    mcp.run(transport="http", host="0.0.0.0", port=8000)
