// Command basic demonstrates a minimal agent with a single tool.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
	"github.com/rashadism/agents-go/pkg/agent"
)

func main() {
	provider, err := openai.New(config.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}

	searchFlights := agent.Tool{
		Name:        "search_flights",
		Description: "Search for available flights between two cities on a given date",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"from": map[string]any{
					"type":        "string",
					"description": "Departure city",
				},
				"to": map[string]any{
					"type":        "string",
					"description": "Arrival city",
				},
				"date": map[string]any{
					"type":        "string",
					"description": "Travel date (YYYY-MM-DD)",
				},
			},
			"required": []string{"from", "to", "date"},
		},
		Execute: func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				From string `json:"from"`
				To   string `json:"to"`
				Date string `json:"date"`
			}
			if err := json.Unmarshal(args, &p); err != nil {
				return "", err
			}
			return fmt.Sprintf(`[{"airline":"SkyLine","flight":"SL-402","from":%q,"to":%q,"date":%q,"depart":"08:30","arrive":"11:45","price":289},{"airline":"AirConnect","flight":"AC-118","from":%q,"to":%q,"date":%q,"depart":"14:15","arrive":"17:30","price":345}]`,
				p.From, p.To, p.Date, p.From, p.To, p.Date), nil
		},
	}

	prompt := "Find me flights from New York to London on 2026-05-15"

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  model:          gpt-5.2")
	fmt.Println("  system prompt:  You are a helpful travel assistant.")
	fmt.Println("  tools:          search_flights")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()

	a, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a helpful travel assistant. Help users find flights and plan trips."),
		agent.WithTools(searchFlights),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("=== Agent Loop ===")
	result, err := a.Run(context.Background(), []providers.Message{
		{Role: providers.RoleUser, Content: prompt},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, msg := range result.Messages {
		switch msg.Role {
		case providers.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					fmt.Printf("  [model] tool call: %s (id=%s) args=%s\n", tc.Function.Name, tc.ID, tc.Function.Arguments)
				}
			}
			if msg.ContentString() != "" {
				fmt.Printf("  [model] response: %s\n", msg.ContentString())
			}
		case providers.RoleTool:
			content := msg.ContentString()
			if len(content) > 200 {
				content = content[:200] + "...[truncated]"
			}
			fmt.Printf("  [tool]  result (id=%s): %s\n", msg.ToolCallID, content)
		}
	}

	fmt.Println()
	fmt.Printf("=== Summary ===\n")
	fmt.Printf("  messages: %d\n", len(result.Messages))
}
