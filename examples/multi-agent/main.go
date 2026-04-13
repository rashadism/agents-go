// Command multi-agent demonstrates agent-as-tool composition where a parent
// orchestrator delegates research to specialist sub-agents that search the web
// using the Tavily API.
//
// Required environment variables:
//
//	OPENAI_API_KEY  — OpenAI API key
//	TAVILY_API_KEY  — Tavily API key (free at https://app.tavily.com)
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
	"github.com/rashadism/agents-go/pkg/agent"
	"github.com/rashadism/agents-go/tools/todos"
)

func main() {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	tavilyKey := os.Getenv("TAVILY_API_KEY")
	if openaiKey == "" || tavilyKey == "" {
		log.Fatal("OPENAI_API_KEY and TAVILY_API_KEY must be set")
	}

	provider, err := openai.New(config.WithAPIKey(openaiKey))
	if err != nil {
		log.Fatal(err)
	}

	// --- Researcher sub-agent ---

	researcher, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a research specialist. Use web_search to find information on the topic you are given. Perform multiple searches if needed. Return a clear, concise summary of your findings with key facts and sources."),
		agent.WithTools(tavilySearch(tavilyKey)),
	)
	if err != nil {
		log.Fatal(err)
	}

	// --- Orchestrator ---

	orchestrator, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a research orchestrator. First, use write_todos to plan your research tasks. Then delegate each task to the researcher sub-agent — launch multiple in parallel when topics are independent. As results come back, update your todos. Once all research is complete, synthesize findings into a well-structured final answer."),
		agent.WithTools(
			todos.NewWriteTodosTool(),
			agent.AgentTool("researcher", "Delegates a focused research task — provide a specific topic to investigate. The researcher will search the web and return a summary.", researcher),
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	// --- Run ---

	prompt := "Compare Rust and Go programming languages — their design philosophy, performance characteristics, and best use cases. Give me a structured comparison."

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  orchestrator:  gpt-5.2")
	fmt.Println("  tools:         write_todos")
	fmt.Println("  sub-agents:")
	fmt.Println("    - researcher (gpt-5.2): web_search (Tavily)")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()
	fmt.Println("=== Agent Loop ===")

	result, err := orchestrator.Run(context.Background(), []providers.Message{
		{Role: providers.RoleUser, Content: prompt},
	})
	if err != nil {
		log.Fatal(err)
	}

	// --- Display ---

	toolCalls := 0
	for _, msg := range result.Messages {
		switch msg.Role {
		case providers.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					toolCalls++
					args := tc.Function.Arguments
					if len(args) > 120 {
						args = args[:120] + "..."
					}
					fmt.Printf("  [orchestrator] tool call: %s (id=%s)\n", tc.Function.Name, tc.ID)
					fmt.Printf("                 args: %s\n", args)
				}
			}
			if content := msg.ContentString(); content != "" {
				fmt.Printf("  [orchestrator] response:\n%s\n", content)
			}
		case providers.RoleTool:
			content := msg.ContentString()
			if len(content) > 300 {
				content = content[:300] + "...[truncated]"
			}
			fmt.Printf("  [sub-agent]    result (id=%s): %s\n", msg.ToolCallID, content)
		}
	}

	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("  messages:    %d\n", len(result.Messages))
	fmt.Printf("  tool calls:  %d (sub-agent delegations)\n", toolCalls)
}
