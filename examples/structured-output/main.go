// Command structured-output demonstrates provider-native structured output.
// The model returns a structured recipe directly via json_schema response format.
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

	strict := true

	prompt := "I have chicken, garlic, and lemons. Give me a recipe."

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  model:          gpt-5.2")
	fmt.Println("  system prompt:  You are a creative chef assistant...")
	fmt.Println("  strategy:       provider (native json_schema response format)")
	fmt.Println("  output schema:  recipe")
	fmt.Println("  strict:         true")
	fmt.Println("  tools:          (none)")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()
	fmt.Println("=== Agent Loop ===")

	a, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a creative chef assistant. Generate recipes based on the ingredients provided."),
		agent.WithStructuredOutput(&agent.StructuredOutput{
			Strategy:    agent.OutputStrategyProvider,
			Name:        "recipe",
			Description: "A structured recipe",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "Name of the recipe",
					},
					"servings": map[string]any{
						"type":        "integer",
						"description": "Number of servings",
					},
					"prep_time_minutes": map[string]any{
						"type":        "integer",
						"description": "Preparation time in minutes",
					},
					"ingredients": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"item":     map[string]any{"type": "string"},
								"quantity": map[string]any{"type": "string"},
							},
							"required":             []string{"item", "quantity"},
							"additionalProperties": false,
						},
					},
					"steps": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
				"required":             []string{"name", "servings", "prep_time_minutes", "ingredients", "steps"},
				"additionalProperties": false,
			},
			Strict: &strict,
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	result, err := a.Run(context.Background(), []providers.Message{
		{Role: providers.RoleUser, Content: prompt},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, msg := range result.Messages {
		if msg.Role == providers.RoleAssistant && msg.ContentString() != "" {
			content := msg.ContentString()
			if len(content) > 200 {
				content = content[:200] + "...[truncated]"
			}
			fmt.Printf("  [model] response: %s\n", content)
		}
	}

	fmt.Println()
	fmt.Printf("=== Summary ===\n")
	fmt.Printf("  messages: %d\n", len(result.Messages))

	if result.StructuredResponse != nil {
		fmt.Println()
		fmt.Println("=== Recipe ===")
		formatted, _ := json.MarshalIndent(json.RawMessage(result.StructuredResponse), "", "  ")
		fmt.Println(string(formatted))
	} else {
		fmt.Println()
		fmt.Println("  [warn] no structured response received")
	}
}
