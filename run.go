package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"io/ioutil"
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
	rootDir  = "."
	maxIters = 12

	anthropicURL = "https://codex.claudebuddy.fun/api"
	modelName    = "anthropic/claude-sonnet-4.5"

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
// Allowed paths
// =========================
func isAllowedPath(rel string) bool {
	if len(rel) >= 7 && rel[:7] == "models/" {
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
// Collect all project files
// =========================
func collectFiles() (map[string]string, error) {
	files := make(map[string]string)

	err := filepath.Walk(rootDir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		rel, _ := filepath.Rel(rootDir, path)
		content, _ := ioutil.ReadFile(path)
		files[rel] = string(content)
		return nil
	})

	return files, err
}

// =========================
// Run training test
// =========================
func runTestMode() (int, string) {
	cmd := exec.Command("venv/bin/python", "train.py", "--mode", "test")
	cmd.Dir = "."

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
// Ask Claude (timeout + logging)
// =========================
func askClaude(files map[string]string, testOutput string) ([]PatchAction, error) {

	prompt, _ := ioutil.ReadFile(promptFile)

	body := map[string]interface{}{
		"model":       modelName,
		"max_tokens":  6000,
		"temperature": 0,
		"system":      string(prompt),
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

	reqBytes, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", anthropicURL, bytes.NewBuffer(reqBytes))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", os.Getenv("ANTHROPIC_AUTH_TOKEN"))
	req.Header.Set("anthropic-version", "2023-06-01")

	// Timeout-enabled HTTP client
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

	respBytes, _ := ioutil.ReadAll(resp.Body)

	fmt.Println("\n====== RAW CLAUDE RESPONSE ======")
	fmt.Println(string(respBytes))
	fmt.Println("=================================\n")

	var msg ClaudeMessageResponse
	if err := json.Unmarshal(respBytes, &msg); err != nil {
		return nil, fmt.Errorf("Failed to parse JSON: %v", err)
	}

	if msg.Content == nil || len(msg.Content) == 0 {
		return nil, fmt.Errorf("Claude returned empty content")
	}

	raw := msg.Content[0].Text

	fmt.Println("\n====== PARSED TEXT ======")
	fmt.Println(raw)
	fmt.Println("==========================\n")

	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}

	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, fmt.Errorf("Failed parsing actions: %v", err)
	}

	return parsed.Actions, nil
}

// =========================
// Apply patch actions
// =========================
func applyActions(actions []PatchAction) error {

	fmt.Println("\n====== APPLYING ACTIONS ======")
	pretty, _ := json.MarshalIndent(actions, "", "  ")
	fmt.Println(string(pretty))
	fmt.Println("===============================\n")

	for _, a := range actions {

		switch a.Op {

		case "write":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("Forbidden write: %s", a.File)
			}
			os.MkdirAll(filepath.Dir(a.File), 0755)
			ioutil.WriteFile(a.File, []byte(a.Content), 0644)

		case "mkdir":
			if !isAllowedPath(a.Dir) {
				return fmt.Errorf("Forbidden mkdir: %s", a.Dir)
			}
			os.MkdirAll(a.Dir, 0755)

		case "delete":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("Forbidden delete: %s", a.File)
			}
			os.Remove(a.File)
		}
	}

	return nil
}

// =========================
// Install Python deps
// =========================
func installRequirements() error {
	cmd := exec.Command("venv/bin/pip", "install", "-r", "requirements.txt")

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
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

// =========================
// MAIN LOOP — FIX FIRST, TEST LATER
// =========================
func main() {

	testOutput := "" // first round no test

	for i := 1; i <= maxIters; i++ {

		fmt.Printf("\n====== ITERATION %d ======\n", i)

		files, _ := collectFiles()

		// 1) Ask Claude FIRST
		actions, err := askClaude(files, testOutput)
		if err != nil {
			fmt.Println("Claude error:", err)
			return
		}

		// 2) Apply fixes
		if err := applyActions(actions); err != nil {
			fmt.Println("Apply error:", err)
			return
		}

		// 3) pip install
		if err := installRequirements(); err != nil {
			fmt.Println("pip install error:", err)
			return
		}

		// 4) Now run test
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
