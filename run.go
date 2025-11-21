package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// =========================
// Config
// =========================
const (
	projectDir = "project"
	maxIters   = 12

	anthropicURL = "https://api.siray.ai/v1/messages"
	// anthropicURL = "https://codex.claudebuddy.fun/api/v1/messages"
	modelName = "anthropic/claude-sonnet-4.5"

	promptFile = "system_prompt.txt"
)

type PatchAction struct {
	Op      string `json:"op"`
	File    string `json:"file,omitempty"`
	Dir     string `json:"dir,omitempty"`
	Content string `json:"content,omitempty"`
}

type ClaudeMessageResponse struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

// =========================
// Allowed paths inside project/
// =========================
func isAllowedPath(rel string) bool {
	if rel == "model.py" {
		return true
	}
	if rel == "train.py" {
		return true
	}
	if rel == "config.py" {
		return true
	}
	if rel == "requirements.txt" {
		return true
	}
	return false
}

// =========================
// Collect project files ONLY
// =========================
// =========================
// Collect ONLY top-level files inside ./project
// =========================
func collectFiles() (map[string]string, error) {
	files := make(map[string]string)

	entries, err := os.ReadDir(projectDir)
	if err != nil {
		return nil, fmt.Errorf("Failed to read project directory: %v", err)
	}

	for _, entry := range entries {

		// ❌ Skip all directories (don't enter subdirectories)
		if entry.IsDir() {
			continue
		}

		// ❌ Optional: skip large files (>200KB)
		info, err := entry.Info()
		if err == nil && info.Size() > 200*1024 {
			fmt.Printf("⚠️ Skipping large file: %s (%d bytes)\n", entry.Name(), info.Size())
			continue
		}

		// Correctly read file content
		path := filepath.Join(projectDir, entry.Name())
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("Failed to read file %s: %v", entry.Name(), err)
		}

		files[entry.Name()] = string(content)
	}

	return files, nil
}

// =========================
// Run training test (cd project/)
// =========================
func runTestMode() (int, string) {
	cmd := exec.Command("venv/bin/python", "train.py", "--mode", "test")
	cmd.Dir = projectDir

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	code := 0
	if err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			code = e.ExitCode()
		} else {
			code = 1
		}
	}

	return code, out.String()
}

// =========================
// Extract JSON from text
// =========================
func extractJSON(text string) string {
	// Try to find JSON between ```json and ``` markers
	start := bytes.Index([]byte(text), []byte("```json"))
	if start != -1 {
		start += 7 // Skip past ```json
		end := bytes.Index([]byte(text[start:]), []byte("```"))
		if end != -1 {
			return text[start : start+end]
		}
	}

	// Try to find JSON between ``` and ``` markers
	start = bytes.Index([]byte(text), []byte("```"))
	if start != -1 {
		start += 3 // Skip past ```
		end := bytes.Index([]byte(text[start:]), []byte("```"))
		if end != -1 {
			candidate := text[start : start+end]
			// Check if it looks like JSON
			candidate = string(bytes.TrimSpace([]byte(candidate)))
			if len(candidate) > 0 && (candidate[0] == '{' || candidate[0] == '[') {
				return string(candidate)
			}
		}
	}

	// Try to find raw JSON (starts with { or [)
	trimmed := bytes.TrimSpace([]byte(text))
	if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') {
		return string(trimmed)
	}

	return ""
}

// =========================
// Ask Claude
// =========================
func askClaude(files map[string]string, testOutput string) ([]PatchAction, error) {

	promptBytes, err := os.ReadFile(promptFile)
	if err != nil {
		return nil, fmt.Errorf("Failed to read prompt file: %v", err)
	}

	body := map[string]interface{}{
		"model":       modelName,
		"max_tokens":  6000,
		"temperature": 0,
		"system":      string(promptBytes),
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]string{
					{"type": "text", "text": "PROJECT FILES:\n" + toJSON(files)},
					{"type": "text", "text": "TEST OUTPUT:\n" + testOutput},
					{"type": "text", "text": "Generate JSON patch actions."},
				},
			},
		},
	}

	reqBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("Failed to marshal request: %v", err)
	}
	req, err := http.NewRequest("POST", anthropicURL, bytes.NewBuffer(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("Failed to create request: %v", err)
	}

	fmt.Println(body)

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", os.Getenv("ANTHROPIC_AUTH_TOKEN"))
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{
		Timeout: 180 * time.Second,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   60 * time.Second,
				KeepAlive: 60 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   30 * time.Second,
			ExpectContinueTimeout: 30 * time.Second,
			IdleConnTimeout:       90 * time.Second,
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   10,
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("Failed to read response: %v", err)
	}

	// Check HTTP status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(respBytes))
	}

	fmt.Println("\n====== RAW CLAUDE RESPONSE ======")
	fmt.Println(string(respBytes))
	fmt.Println("=================================")

	var msg ClaudeMessageResponse
	if err := json.Unmarshal(respBytes, &msg); err != nil {
		return nil, fmt.Errorf("Failed to parse JSON: %v", err)
	}

	if len(msg.Content) == 0 {
		return nil, fmt.Errorf("Claude returned empty content")
	}

	raw := msg.Content[0].Text

	fmt.Println("\n====== PARSED TEXT ======")
	fmt.Println(raw)
	fmt.Println("==========================")

	// Extract JSON from response (handle markdown code blocks or plain text)
	jsonText := extractJSON(raw)
	if jsonText == "" {
		return nil, fmt.Errorf("No JSON found in Claude's response")
	}

	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}

	if err := json.Unmarshal([]byte(jsonText), &parsed); err != nil {
		return nil, fmt.Errorf("Failed parsing actions: %v", err)
	}

	if len(parsed.Actions) == 0 {
		return nil, fmt.Errorf("No actions returned by Claude")
	}

	return parsed.Actions, nil
}

// =========================
// Apply patch actions
// =========================
func applyActions(actions []PatchAction) error {

	fmt.Println("\n====== APPLYING ACTIONS ======")
	pretty, err := json.MarshalIndent(actions, "", "  ")
	if err != nil {
		fmt.Printf("Warning: Failed to format actions: %v\n", err)
	} else {
		fmt.Println(string(pretty))
	}
	fmt.Println("===============================")

	for _, a := range actions {
		switch a.Op {

		case "write":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("Forbidden write: %s", a.File)
			}
			dst := filepath.Join(projectDir, a.File)
			if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
				return fmt.Errorf("Failed to create directory for %s: %v", a.File, err)
			}
			if err := os.WriteFile(dst, []byte(a.Content), 0644); err != nil {
				return fmt.Errorf("Failed to write file %s: %v", a.File, err)
			}

		case "mkdir":
			if !isAllowedPath(a.Dir) {
				return fmt.Errorf("Forbidden mkdir: %s", a.Dir)
			}
			dst := filepath.Join(projectDir, a.Dir)
			if err := os.MkdirAll(dst, 0755); err != nil {
				return fmt.Errorf("Failed to create directory %s: %v", a.Dir, err)
			}

		case "delete":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("Forbidden delete: %s", a.File)
			}
			dst := filepath.Join(projectDir, a.File)
			if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("Failed to delete file %s: %v", a.File, err)
			}
		}
	}

	return nil
}

// =========================
// Install Python deps
// =========================
func installRequirements() error {
	cmd := exec.Command("venv/bin/pip", "install", "-r", "requirements.txt")
	cmd.Dir = projectDir

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	fmt.Println("[pip install] Starting…")
	err := cmd.Run()
	fmt.Println(out.String())

	return err
}

// =========================
// JSON helper
// =========================
func toJSON(v interface{}) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("Error marshaling JSON: %v", err)
	}
	return string(b)
}

// =========================
// MAIN LOOP
// =========================
func main() {

	testOutput := ""

	for i := 1; i <= maxIters; i++ {

		fmt.Printf("\n====== ITERATION %d ======\n", i)

		files, err := collectFiles()

		if err != nil {
			fmt.Println("Collect files error:", err)
			return
		}

		actions, err := askClaude(files, testOutput)
		if err != nil {
			fmt.Println("Claude error:", err)
			return
		}

		if err := applyActions(actions); err != nil {
			fmt.Println("Apply error:", err)
			return
		}

		if err := installRequirements(); err != nil {
			fmt.Println("pip install error:", err)
			return
		}

		exitCode, out := runTestMode()
		testOutput = out

		fmt.Println("====== TEST OUTPUT ======")
		fmt.Println(out)
		fmt.Println("==========================")

		if exitCode == 0 {
			fmt.Println("🎉 All tests passed! Finished.")
			return
		}

		fmt.Println("❌ Test failed, continuing loop…")
	}

	fmt.Println("❌ Max iterations reached without success.")
}
