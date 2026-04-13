// Command streaming-structured combines streaming with structured output
// for a product comparison agent.
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

var comparisonSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"summary": map[string]any{
			"type":        "string",
			"description": "Brief overall comparison summary",
		},
		"products": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name":   map[string]any{"type": "string"},
					"pros":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					"cons":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					"rating": map[string]any{"type": "number", "description": "Rating out of 5"},
				},
				"required": []string{"name", "pros", "cons", "rating"},
			},
		},
		"recommendation": map[string]any{
			"type":        "string",
			"description": "Which product to buy and why",
		},
	},
	"required": []string{"summary", "products", "recommendation"},
}

func main() {
	provider, err := openai.New(config.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}

	getProductSpecs := agent.Tool{
		Name:        "get_product_specs",
		Description: "Get detailed specifications for a product by name",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"product": map[string]any{
					"type":        "string",
					"description": "Product name to look up",
				},
			},
			"required": []string{"product"},
		},
		Execute: func(_ context.Context, args json.RawMessage) (string, error) {
			var p struct {
				Product string `json:"product"`
			}
			json.Unmarshal(args, &p)
			// Return different specs depending on product name keywords
			return fmt.Sprintf(`{"product":%q,"price":299,"battery_hours":30,"weight_g":250,"noise_cancelling":true,"bluetooth":"5.3","driver_size_mm":40,"water_resistant":true,"user_rating":4.3,"reviews":1250}`, p.Product), nil
		},
	}

	prompt := "Compare the Sony WH-1000XM5 and the Bose QuietComfort Ultra headphones."

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  model:           gpt-5.2")
	fmt.Println("  system prompt:   You are a product comparison expert...")
	fmt.Println("  strategy:        tool (structured output via tool call)")
	fmt.Println("  output schema:   comparison_report")
	fmt.Println("  tools:           get_product_specs, comparison_report (structured output)")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()
	fmt.Println("=== Agent Loop ===")

	a, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a product comparison expert. Use the tools to gather specs, then submit a structured comparison report."),
		agent.WithTools(getProductSpecs),
		agent.WithStructuredOutput(&agent.StructuredOutput{
			Strategy:    agent.OutputStrategyTool,
			Name:        "comparison_report",
			Description: "Submit the final structured product comparison",
			Schema:      comparisonSchema,
			HandleErrors: func(err error) string {
				return fmt.Sprintf("Invalid format: %v. Please retry with valid JSON.", err)
			},
		}),
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

	if result.StructuredResponse != nil {
		fmt.Println()
		fmt.Println("=== Comparison Report ===")
		formatted, _ := json.MarshalIndent(json.RawMessage(result.StructuredResponse), "", "  ")
		fmt.Println(string(formatted))
	} else {
		fmt.Println()
		fmt.Println("  [warn] no structured response — model responded with plain text")
		for i := len(result.Messages) - 1; i >= 0; i-- {
			if result.Messages[i].Role == providers.RoleAssistant && result.Messages[i].ContentString() != "" {
				fmt.Println(result.Messages[i].ContentString())
				break
			}
		}
	}
}
