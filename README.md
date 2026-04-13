# agents-go

A Go framework for building LLM agents. Uses [any-llm-go](https://github.com/mozilla-ai/any-llm-go) as the LLM provider abstraction. **Not feature complete yet.**

## Features

- **Agent loop** — model call → tool execution → model call, with parallel tool execution
- **Streaming** — channel-based streaming with text deltas and tool call events
- **Structured output** — provider-native (`json_schema`) or tool-based strategy, with auto-fallback
- **Middleware** — composable hooks for model calls, tool calls, and agent lifecycle
- **Multi-agent** — compose agents as tools for orchestrator/specialist patterns
- **MCP** — connect to MCP servers (streamable HTTP) and use their tools
- **Built-in tools** — task tracking (`write_todos`)

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

### Multi-agent composition

```go
// Create a specialist sub-agent with its own tools and system prompt
researcher, _ := agent.CreateAgent(provider, "gpt-5.2",
    agent.WithSystemPrompt("You are a research specialist."),
    agent.WithTools(webSearch),
)

// Give it to a parent agent as a tool
orchestrator, _ := agent.CreateAgent(provider, "gpt-5.2",
    agent.WithSystemPrompt("You coordinate research tasks."),
    agent.WithTools(
        agent.AgentTool("researcher", "Delegates research tasks", researcher),
    ),
)

result, _ := orchestrator.Run(ctx, messages)
```

## Examples

```bash
go run ./examples/basic                  # Travel assistant — single tool
go run ./examples/streaming              # Travel assistant — streaming with parallel tool calls
go run ./examples/middleware              # Customer support — logging and message-budget middleware
go run ./examples/structured-output      # Recipe generator — provider-native structured output
go run ./examples/streaming-structured   # Product comparison — streaming + structured output
go run ./examples/multi-agent            # Deep research — orchestrator + researcher sub-agents (requires TAVILY_API_KEY)
```
