// Command middleware demonstrates the middleware system with logging and
// message-budget hooks for a customer support agent.
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
)

type lookupOrderInput struct {
	OrderID string `json:"order_id" jsonschema:"The order ID to look up"`
}

type checkInventoryInput struct {
	ProductName string `json:"product_name" jsonschema:"Name of the product"`
}

// loggingMiddleware logs agent lifecycle, model calls, and tool calls.
type loggingMiddleware struct {
	step      int
	toolCalls int
}

func (m *loggingMiddleware) Name() string { return "logging" }

func (m *loggingMiddleware) BeforeAgent(_ context.Context, _ *agent.State) error {
	fmt.Println("  [hook] BeforeAgent")
	return nil
}

func (m *loggingMiddleware) AfterAgent(_ context.Context, state *agent.State) error {
	fmt.Printf("  [hook] AfterAgent (%d messages)\n", len(state.Messages))
	return nil
}

func (m *loggingMiddleware) BeforeModel(_ context.Context, _ *agent.State) error {
	m.step++
	fmt.Printf("  [hook] BeforeModel — iteration %d\n", m.step)
	return nil
}

func (m *loggingMiddleware) AfterModel(_ context.Context, _ *agent.State) error {
	fmt.Printf("  [hook] AfterModel — iteration %d\n", m.step)
	return nil
}

func (m *loggingMiddleware) WrapModelCall(ctx context.Context, req *agent.ModelRequest, next agent.ModelCallHandler) (*agent.ModelResponse, error) {
	fmt.Printf("  [mw] WrapModelCall — %d messages, %d tools available\n", len(req.Messages), len(req.Tools))
	resp, err := next(ctx, req)
	if err != nil {
		fmt.Printf("  [mw] WrapModelCall error: %v\n", err)
	} else if len(resp.Message.ToolCalls) > 0 {
		fmt.Printf("  [mw] WrapModelCall — model returned %d tool call(s)\n", len(resp.Message.ToolCalls))
	} else {
		text := resp.Message.ContentString()
		if len(text) > 100 {
			text = text[:100] + "..."
		}
		fmt.Printf("  [mw] WrapModelCall — model returned text: %s\n", text)
	}
	return resp, err
}

func (m *loggingMiddleware) WrapToolCall(ctx context.Context, req *agent.ToolCallRequest, next agent.ToolCallHandler) (*agent.ToolCallResponse, error) {
	m.toolCalls++
	fmt.Printf("  [mw] WrapToolCall #%d — %s (id=%s) args=%s\n", m.toolCalls, req.ToolCall.Function.Name, req.ToolCall.ID, req.ToolCall.Function.Arguments)
	resp, err := next(ctx, req)
	if err != nil {
		fmt.Printf("  [mw] WrapToolCall — %s error: %v\n", req.ToolCall.Function.Name, err)
	} else {
		content := resp.Content
		if len(content) > 150 {
			content = content[:150] + "...[truncated]"
		}
		fmt.Printf("  [mw] WrapToolCall — %s result: %s\n", req.ToolCall.Function.Name, content)
	}
	return resp, err
}

// messageBudgetMiddleware stops the agent after N model iterations,
// preventing runaway conversations.
type messageBudgetMiddleware struct {
	max   int
	count int
}

func (m *messageBudgetMiddleware) Name() string { return "message-budget" }

func (m *messageBudgetMiddleware) BeforeModel(_ context.Context, state *agent.State) error {
	m.count++
	if m.count > m.max {
		fmt.Printf("  [hook] message-budget: reached limit of %d iterations, stopping\n", m.max)
		state.Done = true
	}
	return nil
}

func main() {
	provider, err := openai.New(config.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}

	lookupOrder := agent.NewTool("lookup_order",
		"Look up a customer order by order ID",
		func(_ context.Context, in lookupOrderInput) (string, error) {
			return fmt.Sprintf(`{"order_id":%q,"status":"shipped","items":[{"name":"Wireless Headphones","qty":1,"price":79.99}],"tracking":"TRK-98765","estimated_delivery":"2026-04-08"}`, in.OrderID), nil
		})

	checkInventory := agent.NewTool("check_inventory",
		"Check if a product is currently in stock",
		func(_ context.Context, in checkInventoryInput) (string, error) {
			return fmt.Sprintf(`{"product":%q,"in_stock":true,"quantity_available":23,"warehouse":"US-West"}`, in.ProductName), nil
		})

	prompt := "I ordered some wireless headphones (order #ORD-1234) but want to add a charging case. Is it in stock? And where's my order?"

	fmt.Println("=== Agent Configuration ===")
	fmt.Println("  model:          gpt-5.2")
	fmt.Println("  system prompt:  You are a friendly customer support agent...")
	fmt.Println("  tools:          lookup_order, check_inventory")
	fmt.Println("  middleware:     logging (hooks + model/tool wrapping), message-budget (limit=5)")
	fmt.Println()
	fmt.Println("=== Input ===")
	fmt.Printf("  user: %s\n", prompt)
	fmt.Println()
	fmt.Println("=== Agent Loop (with middleware logging) ===")

	a, err := agent.CreateAgent(provider, "gpt-5.2",
		agent.WithSystemPrompt("You are a friendly customer support agent. Help customers with their orders and product questions. Be concise and helpful."),
		agent.WithTools(lookupOrder, checkInventory),
		agent.WithMiddleware(
			&loggingMiddleware{},
			&messageBudgetMiddleware{max: 5},
		),
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

	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Printf("  messages: %d\n", len(result.Messages))
	fmt.Println()
	fmt.Println("=== Final Response ===")
	for i := len(result.Messages) - 1; i >= 0; i-- {
		msg := result.Messages[i]
		if msg.Role == providers.RoleAssistant && msg.ContentString() != "" {
			fmt.Println(msg.ContentString())
			break
		}
	}
}
