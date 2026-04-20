package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// NewToolFromSchema builds a Tool from an explicit JSON Schema.
//
// Use this when the schema cannot be expressed as a static Go struct: tools
// converted from MCP servers, schemas with enum constraints, or any case
// where the schema is generated or loaded at runtime. For static schemas,
// prefer [NewTool], which derives the schema from the input type.
//
// The execute function receives raw JSON arguments — it must perform its own
// unmarshalling. NewToolFromSchema does not normalize or validate the schema;
// the caller is responsible for providing a schema the LLM provider accepts.
func NewToolFromSchema(
	name, description string,
	parameters map[string]any,
	execute func(ctx context.Context, arguments json.RawMessage) (string, error),
) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  parameters,
		Execute:     execute,
	}
}

// NewTool builds a strongly-typed Tool from an input struct.
//
// The JSON Schema for T is generated from its fields via reflection. Use
// struct tags to control the schema:
//   - `json:"name"`           — property name in the schema
//   - `json:"name,omitempty"` — same, but the field is not required
//   - `jsonschema:"text"`     — field description shown to the model
//
// Example:
//
//	type SearchInput struct {
//	    Query string `json:"query" jsonschema:"the search query"`
//	    Limit int    `json:"limit,omitempty" jsonschema:"max results"`
//	}
//
//	tool := agent.NewTool("search", "Search the corpus.",
//	    func(ctx context.Context, in SearchInput) (string, error) {
//	        return doSearch(in.Query, in.Limit), nil
//	    })
//
// NewTool panics if T's schema cannot be generated. T is statically known
// at compile time, so a schema failure is a programmer error and should
// surface at startup rather than at request time.
func NewTool[T any](name, description string, fn func(context.Context, T) (string, error)) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  mustGenerateSchema[T](name),
		Execute: func(ctx context.Context, raw json.RawMessage) (string, error) {
			var in T
			if err := json.Unmarshal(raw, &in); err != nil {
				return "", fmt.Errorf("tool %q: invalid arguments: %w", name, err)
			}
			return fn(ctx, in)
		},
	}
}

// mustGenerateSchema reflects T into a provider-friendly JSON Schema map.
// It panics on failure (see NewTool).
func mustGenerateSchema[T any](toolName string) map[string]any {
	s, err := jsonschema.For[T](nil)
	if err != nil {
		panic(fmt.Sprintf("agent.NewTool[%s]: schema generation failed: %v", toolName, err))
	}

	raw, err := json.Marshal(s)
	if err != nil {
		panic(fmt.Sprintf("agent.NewTool[%s]: marshal schema failed: %v", toolName, err))
	}

	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		panic(fmt.Sprintf("agent.NewTool[%s]: unmarshal schema failed: %v", toolName, err))
	}

	normalizeToolSchema(m)
	return m
}

// normalizeToolSchema adjusts a generated schema for LLM tool-call APIs:
//   - Ensures the root has "type": "object" (required by OpenAI tool calls).
//   - Replaces additionalProperties: {"not": {}} with additionalProperties: false,
//     which is the form providers expect.
func normalizeToolSchema(m map[string]any) {
	if _, ok := m["type"]; !ok {
		m["type"] = "object"
	}
	if ap, ok := m["additionalProperties"].(map[string]any); ok {
		if not, ok := ap["not"].(map[string]any); ok && len(not) == 0 {
			m["additionalProperties"] = false
		}
	}
}
