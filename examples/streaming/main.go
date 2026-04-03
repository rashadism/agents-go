// Command streaming demonstrates the agent's streaming API with real-time
// text deltas and tool call events.
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
	"github.com/rashadism/agents-go"
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
				"from": map[string]any{"type": "string", "description": "Departure city"},
				"to":   map[string]any{"type": "string", "description": "Arrival city"},
				"date": map[string]any{"type": "string", "description": "Travel date (YYYY-MM-DD)"},
			},
			"required": []string{"from", "to", "date"},
		},
		Execute: func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				From string `json:"from"`
				To   string `json:"to"`
				Date string `json:"date"`
			}
			json.Unmarshal(args, &p)
			return fmt.Sprintf(`[{"airline":"SkyLine","flight":"SL-402","from":%q,"to":%q,"date":%q,"depart":"08:30","arrive":"11:45","price":289},{"airline":"AirConnect","flight":"AC-118","from":%q,"to":%q,"date":%q,"depart":"14:15","arrive":"17:30","price":345}]`,
				p.From, p.To, p.Date, p.From, p.To, p.Date), nil
		},
	}

	searchHotels := agent.Tool{
		Name:        "search_hotels",
		Description: "Search for available hotels in a city for given dates",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city":      map[string]any{"type": "string", "description": "City name"},
				"checkin":   map[string]any{"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
				"checkout":  map[string]any{"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
			},
			"required": []string{"city", "checkin", "checkout"},
		},
		Execute: func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				City string `json:"city"`
			}
			json.Unmarshal(args, &p)
			return fmt.Sprintf(`[{"name":"The Grand %s","rating":4.5,"price_per_night":180,"amenities":["wifi","pool","gym"]},{"name":"%s Central Inn","rating":4.2,"price_per_night":120,"amenities":["wifi","breakfast"]}]`,
				p.City, p.City), nil
		},
	}

	prompt := "Plan a trip from San Francisco to Tokyo, May 20-25. Find flights and hotels."

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  model:          gpt-5.2")
	fmt.Println("  system prompt:  You are a helpful travel planning assistant.")
	fmt.Println("  tools:          search_flights, search_hotels")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()
	fmt.Println("=== Agent Loop (streaming) ===")

	a, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a helpful travel planning assistant. Search for flights and hotels to help users plan their trips."),
		agent.WithTools(searchFlights, searchHotels),
	)
	if err != nil {
		log.Fatal(err)
	}

	events, errs := a.Stream(context.Background(), []providers.Message{
		{Role: providers.RoleUser, Content: prompt},
	})

	var result *agent.Result
	toolCalls := 0
	iteration := 0
	inModel := false

	for event := range events {
		switch event.Type {
		case agent.StreamEventTextDelta:
			if !inModel {
				iteration++
				fmt.Printf("\n[iteration %d] model responding (text)...\n  ", iteration)
				inModel = true
			}
			fmt.Print(event.Delta)
		case agent.StreamEventToolCallStart:
			if !inModel {
				iteration++
				fmt.Printf("\n[iteration %d] model responding (tool calls)...\n", iteration)
				inModel = true
			}
			toolCalls++
			fmt.Printf("  [tool-call #%d] %s (id=%s)\n", toolCalls, event.ToolName, event.ToolCallID)
			if event.Args != "" {
				fmt.Printf("    args: %s\n", event.Args)
			}
		case agent.StreamEventToolResult:
			inModel = false
			content := event.Content
			if len(content) > 300 {
				content = content[:300] + "...[truncated]"
			}
			fmt.Printf("  [tool-result] %s (id=%s)\n", event.ToolName, event.ToolCallID)
			fmt.Printf("    result: %s\n", content)
		case agent.StreamEventComplete:
			inModel = false
			result = event.Result
		}
	}
	if err := <-errs; err != nil {
		log.Fatal(err)
	}

	fmt.Println()
	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("  iterations:  %d\n", iteration)
	fmt.Printf("  tool calls:  %d\n", toolCalls)
	fmt.Printf("  messages:    %d\n", len(result.Messages))
}
