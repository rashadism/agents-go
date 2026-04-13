package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/rashadism/agents-go/pkg/agent"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Server configures a connection to an MCP server.
type Server struct {
	// Name identifies this server (used for logging and diagnostics).
	Name string
	// URL is the streamable HTTP endpoint (e.g., "http://observer:8080/mcp").
	URL string
	// HTTPClient is an optional custom HTTP client for this server.
	// Use this to configure authentication, timeouts, TLS, etc.
	// If nil, http.DefaultClient is used.
	HTTPClient *http.Client
}

// Client manages connections to multiple MCP servers and converts their
// tools into agent.Tool instances that can be passed to agent.CreateAgent.
//
// Usage:
//
//	client := mcp.NewClient([]mcp.Server{
//	    {Name: "observer", URL: "http://observer:8080/mcp"},
//	})
//	defer client.Close()
//	if err := client.Connect(ctx); err != nil { ... }
//	tools, err := client.Tools(ctx)
//	a, err := agent.CreateAgent(provider, model, agent.WithTools(tools...))
type Client struct {
	servers  []Server
	client   *mcpsdk.Client
	sessions map[string]*mcpsdk.ClientSession
	logger   *slog.Logger
}

// ClientOption configures a Client.
type ClientOption func(*Client)

// WithLogger sets the logger for the MCP client.
func WithLogger(logger *slog.Logger) ClientOption {
	return func(c *Client) { c.logger = logger }
}

// NewClient creates a new MCP client for the given servers.
// Call Connect() to establish sessions before calling Tools().
func NewClient(servers []Server, opts ...ClientOption) *Client {
	c := &Client{
		servers:  servers,
		sessions: make(map[string]*mcpsdk.ClientSession),
		logger:   slog.Default(),
	}
	for _, opt := range opts {
		opt(c)
	}
	c.client = mcpsdk.NewClient(&mcpsdk.Implementation{
		Name:    "agents-go/mcp",
		Version: "1.0.0",
	}, nil)
	return c
}

// Connect establishes sessions with all configured MCP servers.
// If any connection fails, already-established sessions are closed.
func (c *Client) Connect(ctx context.Context) error {
	for _, server := range c.servers {
		transport := &mcpsdk.StreamableClientTransport{
			Endpoint:             server.URL,
			HTTPClient:           server.HTTPClient,
			DisableStandaloneSSE: true,
		}

		session, err := c.client.Connect(ctx, transport, nil)
		if err != nil {
			_ = c.Close()
			return fmt.Errorf("connecting to MCP server %q at %s: %w", server.Name, server.URL, err)
		}

		c.sessions[server.Name] = session
		c.logger.InfoContext(ctx, "connected to MCP server", "server", server.Name, "url", server.URL)
	}
	return nil
}

// Close closes all MCP sessions. Safe to call multiple times.
func (c *Client) Close() error {
	var firstErr error
	for name, session := range c.sessions {
		if err := session.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("closing MCP session %q: %w", name, err)
		}
	}
	c.sessions = make(map[string]*mcpsdk.ClientSession)
	return firstErr
}

// Tools lists tools from all connected MCP servers and returns them as
// agent.Tool instances. Each tool's Execute function forwards calls to
// the originating MCP server session.
func (c *Client) Tools(ctx context.Context) ([]agent.Tool, error) {
	var tools []agent.Tool

	for name, session := range c.sessions {
		mcpTools, err := c.listAllTools(ctx, name, session)
		if err != nil {
			return nil, err
		}

		for _, mt := range mcpTools {
			tool, err := convertMCPTool(session, mt)
			if err != nil {
				return nil, fmt.Errorf("converting tool %q from server %q: %w", mt.Name, name, err)
			}
			tools = append(tools, tool)
		}

		c.logger.InfoContext(ctx, "loaded tools from MCP server", "server", name, "count", len(mcpTools))
	}

	return tools, nil
}

// listAllTools fetches all tools from a session, handling pagination.
func (c *Client) listAllTools(ctx context.Context, serverName string, session *mcpsdk.ClientSession) ([]*mcpsdk.Tool, error) {
	result, err := session.ListTools(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("listing tools from %q: %w", serverName, err)
	}

	tools := result.Tools
	for result.NextCursor != "" {
		result, err = session.ListTools(ctx, &mcpsdk.ListToolsParams{
			Cursor: result.NextCursor,
		})
		if err != nil {
			return nil, fmt.Errorf("listing tools from %q (paginated): %w", serverName, err)
		}
		tools = append(tools, result.Tools...)
	}

	return tools, nil
}

// convertMCPTool converts a single MCP tool to an agent.Tool.
func convertMCPTool(session *mcpsdk.ClientSession, mt *mcpsdk.Tool) (agent.Tool, error) {
	params, err := toParameterMap(mt.InputSchema)
	if err != nil {
		return agent.Tool{}, fmt.Errorf("converting input schema: %w", err)
	}

	return agent.Tool{
		Name:        mt.Name,
		Description: mt.Description,
		Parameters:  params,
		Execute:     mcpToolExecutor(session, mt.Name),
	}, nil
}

// mcpToolExecutor returns an Execute function that forwards calls to the
// MCP server via the given session.
func mcpToolExecutor(session *mcpsdk.ClientSession, toolName string) func(context.Context, json.RawMessage) (string, error) {
	return func(ctx context.Context, args json.RawMessage) (string, error) {
		var arguments map[string]any
		if len(args) > 0 {
			if err := json.Unmarshal(args, &arguments); err != nil {
				return "", fmt.Errorf("invalid tool arguments: %w", err)
			}
		}

		result, err := session.CallTool(ctx, &mcpsdk.CallToolParams{
			Name:      toolName,
			Arguments: arguments,
		})
		if err != nil {
			return "", fmt.Errorf("tool %q: %w", toolName, err)
		}

		text := mcpContentToString(result.Content)
		if result.IsError {
			return "", fmt.Errorf("%s", text)
		}

		return text, nil
	}
}

// mcpContentToString extracts text from MCP content blocks.
func mcpContentToString(content []mcpsdk.Content) string {
	var parts []string
	for _, c := range content {
		if v, ok := c.(*mcpsdk.TextContent); ok {
			parts = append(parts, v.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// toParameterMap converts an MCP input schema (any) to map[string]any.
func toParameterMap(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	if m, ok := schema.(map[string]any); ok {
		return m, nil
	}
	// Fallback: marshal/unmarshal for typed schemas.
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}
