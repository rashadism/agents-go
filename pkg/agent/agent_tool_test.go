package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentTool_BasicDelegation(t *testing.T) {
	t.Parallel()

	// Child agent simply responds with text.
	childProvider := newFakeProvider(completion("The answer is 42"))
	child, err := CreateAgent(childProvider, "child-model",
		WithSystemPrompt("You are a helpful assistant."),
	)
	require.NoError(t, err)

	// Parent agent calls the child via tool, then responds.
	tool := AgentTool("researcher", "Delegates research tasks to a specialist", child)

	parentProvider := newFakeProvider(
		// First call: parent decides to use the researcher tool.
		completion("", toolCall("tc1", "researcher", `{"task":"What is the meaning of life?"}`)),
		// Second call: parent synthesizes the child's result.
		completion("Based on my research, the answer is 42."),
	)
	parent, err := CreateAgent(parentProvider, "parent-model",
		WithTools(tool),
	)
	require.NoError(t, err)

	result, err := parent.Run(context.Background(), []providers.Message{userMsg("What is the meaning of life?")})
	require.NoError(t, err)

	// The tool result message should contain the child's response.
	var toolResultContent string
	for _, msg := range result.Messages {
		if msg.Role == providers.RoleTool {
			toolResultContent = msg.ContentString()
		}
	}
	assert.Equal(t, "The answer is 42", toolResultContent)

	// Final message should be the parent's synthesis.
	last := result.Messages[len(result.Messages)-1]
	assert.Equal(t, "Based on my research, the answer is 42.", last.ContentString())

	// Child should have received the task as a user message.
	require.Len(t, childProvider.calls, 1)
	// Find the user message in the child's call (skip system prompt).
	var childUserMsg string
	for _, msg := range childProvider.calls[0].Messages {
		if msg.Role == providers.RoleUser {
			childUserMsg = msg.ContentString()
		}
	}
	assert.Equal(t, "What is the meaning of life?", childUserMsg)
}

func TestAgentTool_ChildWithTools(t *testing.T) {
	t.Parallel()

	// Child agent uses a tool, then responds.
	childProvider := newFakeProvider(
		completion("", toolCall("tc1", "lookup", `{"q":"test"}`)),
		completion("Found: test result"),
	)
	child, err := CreateAgent(childProvider, "child-model",
		WithTools(echoTool("lookup")),
	)
	require.NoError(t, err)

	tool := AgentTool("searcher", "Searches for information", child)

	// Execute the tool directly.
	result, err := tool.Execute(context.Background(), json.RawMessage(`{"task":"find test info"}`))
	require.NoError(t, err)
	assert.Equal(t, "Found: test result", result)
}

func TestAgentTool_ChildStructuredOutput(t *testing.T) {
	t.Parallel()

	// Child agent returns structured output via tool strategy.
	childProvider := newFakeProvider(
		completion("", toolCall("tc1", "result", `{"answer":42}`)),
	)
	child, err := CreateAgent(childProvider, "child-model",
		WithStructuredOutput(&StructuredOutput{
			Strategy: OutputStrategyTool,
			Name:     "result",
			Schema:   map[string]any{"type": "object"},
		}),
	)
	require.NoError(t, err)

	tool := AgentTool("analyst", "Analyzes data", child)

	result, err := tool.Execute(context.Background(), json.RawMessage(`{"task":"analyze this"}`))
	require.NoError(t, err)
	assert.Equal(t, `{"answer":42}`, result)
}

func TestAgentTool_ChildError(t *testing.T) {
	t.Parallel()

	childProvider := &errProvider{err: errors.New("model unavailable")}
	child, err := CreateAgent(childProvider, "child-model")
	require.NoError(t, err)

	tool := AgentTool("failing", "Always fails", child)

	_, err = tool.Execute(context.Background(), json.RawMessage(`{"task":"do something"}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sub-agent")
	assert.Contains(t, err.Error(), "model unavailable")
}

func TestAgentTool_EmptyTask(t *testing.T) {
	t.Parallel()

	child, err := CreateAgent(newFakeProvider(), "m")
	require.NoError(t, err)

	tool := AgentTool("test", "test", child)

	_, err = tool.Execute(context.Background(), json.RawMessage(`{"task":""}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "task must not be empty")
}

func TestAgentTool_InvalidJSON(t *testing.T) {
	t.Parallel()

	child, err := CreateAgent(newFakeProvider(), "m")
	require.NoError(t, err)

	tool := AgentTool("test", "test", child)

	_, err = tool.Execute(context.Background(), json.RawMessage(`{bad json}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid agent tool arguments")
}

func TestAgentTool_ChildSystemPromptIsolation(t *testing.T) {
	t.Parallel()

	// Child has its own system prompt — parent's prompt should not leak.
	childProvider := newFakeProvider(completion("child response"))
	child, err := CreateAgent(childProvider, "child-model",
		WithSystemPrompt("You are a specialist."),
	)
	require.NoError(t, err)

	tool := AgentTool("specialist", "A specialist sub-agent", child)

	_, err = tool.Execute(context.Background(), json.RawMessage(`{"task":"do your thing"}`))
	require.NoError(t, err)

	// Verify the child's system prompt was used (first message in the call).
	require.Len(t, childProvider.calls, 1)
	msgs := childProvider.calls[0].Messages
	require.True(t, len(msgs) >= 2)
	assert.Equal(t, providers.RoleSystem, msgs[0].Role)
	assert.Equal(t, "You are a specialist.", msgs[0].ContentString())
}

func TestExtractResult_PreferStructured(t *testing.T) {
	t.Parallel()

	result := &Result{
		Messages: []providers.Message{
			{Role: providers.RoleAssistant, Content: "text response"},
		},
		StructuredResponse: json.RawMessage(`{"key":"value"}`),
	}

	assert.Equal(t, `{"key":"value"}`, extractResult(result))
}

func TestExtractResult_FallbackToAssistant(t *testing.T) {
	t.Parallel()

	result := &Result{
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "hello"},
			{Role: providers.RoleAssistant, Content: "world"},
		},
	}

	assert.Equal(t, "world", extractResult(result))
}

func TestExtractResult_EmptyMessages(t *testing.T) {
	t.Parallel()

	result := &Result{Messages: []providers.Message{}}
	assert.Equal(t, "", extractResult(result))
}
