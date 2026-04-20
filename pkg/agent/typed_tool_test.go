package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type searchInput struct {
	Query string `json:"query" jsonschema:"the search query"`
	Limit int    `json:"limit,omitempty" jsonschema:"max number of results"`
}

func TestNewTool_SchemaShape(t *testing.T) {
	t.Parallel()

	tool := NewTool("search", "Search the corpus.",
		func(_ context.Context, _ searchInput) (string, error) { return "", nil })

	assert.Equal(t, "search", tool.Name)
	assert.Equal(t, "Search the corpus.", tool.Description)

	assert.Equal(t, "object", tool.Parameters["type"], `root must be type:"object"`)

	props, ok := tool.Parameters["properties"].(map[string]any)
	require.True(t, ok, "properties must exist and be an object")

	query, ok := props["query"].(map[string]any)
	require.True(t, ok, "query property must exist")
	assert.Equal(t, "string", query["type"])
	assert.Equal(t, "the search query", query["description"], "jsonschema tag must become description")

	limit, ok := props["limit"].(map[string]any)
	require.True(t, ok, "limit property must exist")
	assert.Equal(t, "integer", limit["type"])

	required, ok := tool.Parameters["required"].([]any)
	require.True(t, ok, "required must exist")
	assert.Contains(t, required, "query", "non-omitempty fields must be required")
	assert.NotContains(t, required, "limit", "omitempty fields must be optional")

	// additionalProperties should be normalized to false (not Google's {"not":{}}).
	assert.Equal(t, false, tool.Parameters["additionalProperties"],
		"additionalProperties must be normalized to false for provider compatibility")
}

func TestNewTool_ExecuteUnmarshalsArgs(t *testing.T) {
	t.Parallel()

	var got searchInput
	tool := NewTool("search", "desc",
		func(_ context.Context, in searchInput) (string, error) {
			got = in
			return "ok", nil
		})

	out, err := tool.Execute(context.Background(), json.RawMessage(`{"query":"hello","limit":5}`))
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
	assert.Equal(t, "hello", got.Query)
	assert.Equal(t, 5, got.Limit)
}

func TestNewTool_ExecuteRejectsInvalidJSON(t *testing.T) {
	t.Parallel()

	tool := NewTool("search", "desc",
		func(_ context.Context, _ searchInput) (string, error) { return "", nil })

	_, err := tool.Execute(context.Background(), json.RawMessage(`not-json`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), `tool "search"`)
	assert.Contains(t, err.Error(), "invalid arguments")
}

func TestNewTool_ExecutePropagatesFnError(t *testing.T) {
	t.Parallel()

	tool := NewTool("search", "desc",
		func(_ context.Context, _ searchInput) (string, error) {
			return "", assert.AnError
		})

	_, err := tool.Execute(context.Background(), json.RawMessage(`{"query":"x"}`))
	require.ErrorIs(t, err, assert.AnError)
}

// TestNewTool_IntegratesWithAgentLoop drives an agent end-to-end with a typed
// tool to confirm dispatch/unmarshal works through the real loop.
func TestNewTool_IntegratesWithAgentLoop(t *testing.T) {
	t.Parallel()

	var captured searchInput
	tool := NewTool("search", "desc",
		func(_ context.Context, in searchInput) (string, error) {
			captured = in
			return `{"hits":1}`, nil
		})

	provider := newFakeProvider(
		completion("", toolCall("call_1", "search", `{"query":"agents","limit":3}`)),
		completion("done"),
	)

	a, err := CreateAgent(provider, "model", WithTools(tool))
	require.NoError(t, err)

	result, err := a.Run(context.Background(), []providers.Message{userMsg("go")})
	require.NoError(t, err)

	assert.Equal(t, "agents", captured.Query)
	assert.Equal(t, 3, captured.Limit)

	// Last assistant message is "done".
	last := result.Messages[len(result.Messages)-1]
	assert.Equal(t, providers.RoleAssistant, last.Role)
	assert.Equal(t, "done", last.ContentString())
}
