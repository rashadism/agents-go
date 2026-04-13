package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mozilla-ai/any-llm-go/providers"
)

// AgentTool wraps a child Agent as a Tool that can be used by a parent Agent.
// The parent model calls this tool with a task description, and the child agent
// runs autonomously to complete it, returning the result.
//
// The child agent runs with an isolated context — it receives only the task
// as a user message. Its own system prompt, tools, and middleware apply.
func AgentTool(name, description string, child *Agent) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"task": map[string]any{
					"type":        "string",
					"description": "The task or question for the sub-agent to handle",
				},
			},
			"required": []string{"task"},
		},
		Execute: func(ctx context.Context, arguments json.RawMessage) (string, error) {
			var args struct {
				Task string `json:"task"`
			}
			if err := json.Unmarshal(arguments, &args); err != nil {
				return "", fmt.Errorf("invalid agent tool arguments: %w", err)
			}
			if args.Task == "" {
				return "", fmt.Errorf("task must not be empty")
			}

			result, err := child.Run(ctx, []providers.Message{
				{Role: providers.RoleUser, Content: args.Task},
			})
			if err != nil {
				return "", fmt.Errorf("sub-agent %q: %w", name, err)
			}

			return extractResult(result), nil
		},
	}
}

// extractResult returns the most useful string from an agent Result:
// structured response JSON if present, otherwise the last assistant message.
func extractResult(result *Result) string {
	if result.StructuredResponse != nil {
		return string(result.StructuredResponse)
	}

	// Walk backwards to find the last assistant message with content.
	for i := len(result.Messages) - 1; i >= 0; i-- {
		msg := result.Messages[i]
		if msg.Role == providers.RoleAssistant {
			if content := msg.ContentString(); content != "" {
				return content
			}
		}
	}

	return ""
}
