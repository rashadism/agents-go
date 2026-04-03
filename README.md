# agent

A barebones Go framework for building LLM agents, inspired by langchain patterns. Uses [any-llm-go](https://github.com/mozilla-ai/any-llm-go) as the LLM provider abstraction. **Not feature complete yet.**

## Features

- **Agent loop** — model call → tool execution → model call, with parallel tool execution
- **Streaming** — channel-based streaming with text deltas and tool call events
- **Structured output** — provider-native (`json_schema`) or tool-based strategy, with auto-fallback
- **Middleware** — composable hooks for model calls, tool calls, and agent lifecycle
- **MCP adapter** — connect to MCP servers (streamable HTTP transport) and use their tools

## Install

```bash
go get github.com/rashadism/agents-go
```

## Quick start

```go
a, _ := agent.CreateAgent(provider, "gpt-5.2",
    agent.WithSystemPrompt("You are a helpful assistant."),
    agent.WithTools(myTool),
)

result, _ := a.Run(ctx, []providers.Message{
    {Role: providers.RoleUser, Content: "Hello"},
})
```

## Examples

```bash
go run ./examples/basic                  # Travel planner — single tool
go run ./examples/streaming              # Travel planner — streaming with parallel tool calls
go run ./examples/structured-output      # Recipe generator — provider-native structured output
go run ./examples/middleware             # Customer support — logging and message-budget middleware
go run ./examples/streaming-structured   # Product comparison — streaming + structured output
```
