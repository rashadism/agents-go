package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/rashadism/agents-go/pkg/agent"
)

type webSearchInput struct {
	Query string `json:"query" jsonschema:"The search query"`
}

// tavilySearch returns a web search tool powered by the Tavily API.
func tavilySearch(apiKey string) agent.Tool {
	return agent.NewTool("web_search",
		"Search the web for current information on any topic. Returns relevant results with titles, URLs, and content snippets.",
		func(ctx context.Context, in webSearchInput) (string, error) {
			body, _ := json.Marshal(map[string]any{
				"query":          in.Query,
				"search_depth":   "basic",
				"max_results":    5,
				"include_answer": true,
			})

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.tavily.com/search", bytes.NewReader(body))
			if err != nil {
				return "", err
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+apiKey)

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return "", fmt.Errorf("search failed: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return "", fmt.Errorf("search returned status %d", resp.StatusCode)
			}

			var result struct {
				Answer  string `json:"answer"`
				Results []struct {
					Title   string `json:"title"`
					URL     string `json:"url"`
					Content string `json:"content"`
				} `json:"results"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				return "", fmt.Errorf("parsing response: %w", err)
			}

			var b strings.Builder
			if result.Answer != "" {
				fmt.Fprintf(&b, "Answer: %s\n\n", result.Answer)
			}
			if len(result.Results) > 0 {
				b.WriteString("Results:\n")
				for _, r := range result.Results {
					fmt.Fprintf(&b, "- %s\n  %s\n  %s\n", r.Title, r.URL, r.Content)
				}
			}
			if b.Len() == 0 {
				return "No results found.", nil
			}
			return b.String(), nil
		})
}
