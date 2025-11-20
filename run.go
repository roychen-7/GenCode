package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
)

const (
	rootDir      = "project"
	maxIters     = 12
	anthropicURL = "https://api.siray.ai/v1/messages"
	modelName    = "anthropic/claude-sonnet-4.5"
	promptFile   = "system_prompt.txt"
)

/**************
 * Data Models
 **************/

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

/********************************
 * Allowed path restriction layer
 ********************************/

func isAllowedPath(rel string) bool {

	// allow models/**
	if len(rel) >= 7 && rel[:7] == "models/" {
		return true
	}

	// allow main training script
	if rel == "train.py" {
		return true
	}

	// allow config
	if rel == "config.py" {
		return true
	}

	return false
}

/*************************
 * Project file collectors
 *************************/

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

		// Only collect allowed files
		if !isAllowedPath(rel) {
			return nil
		}

		content, e := ioutil.ReadFile(path)
		if e != nil {
			return e
		}
		files[rel] = string(content)

		return nil
	})

	return files, err
}

/**********************
 * Run pytest tiny test
 **********************/

func runTests() (int, string) {
	pythonPath := "project/venv/bin/python"

	cmd := exec.Command(pythonPath, "train.py", "--mode", "test")
	cmd.Dir = "project" // 🔥 关键点：切换工作目录

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()

	exit := 0
	if err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			exit = e.ExitCode()
		} else {
			exit = 1
		}
	}

	return exit, out.String()
}

/********************
 * Ask Claude for fix
 ********************/

func askClaude(projectFiles map[string]string, testOutput string) ([]PatchAction, error) {

	promptBytes, err := ioutil.ReadFile(promptFile)
	if err != nil {
		return nil, fmt.Errorf("failed reading system prompt: %w", err)
	}
	systemPrompt := string(promptBytes)

	body := map[string]interface{}{
		"model":       modelName,
		"max_tokens":  6000,
		"temperature": 0,
		"system":      systemPrompt,
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]string{
					{"type": "text", "text": "PROJECT FILES:\n" + toJSON(projectFiles)},
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

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, _ := ioutil.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		fmt.Println("HTTP Status:", resp.StatusCode)
		fmt.Println("Response:", string(respBytes))
		return nil, fmt.Errorf("API request failed with status %d", resp.StatusCode)
	}

	var msg ClaudeMessageResponse
	if err := json.Unmarshal(respBytes, &msg); err != nil {
		fmt.Println("RAW:", string(respBytes))
		return nil, fmt.Errorf("Claude response not JSON")
	}

	if len(msg.Content) == 0 {
		fmt.Println("RAW:", string(respBytes))
		return nil, fmt.Errorf("Claude response has no content")
	}

	raw := msg.Content[0].Text

	// Strip markdown code blocks if present
	jsonStr := raw
	if len(raw) > 7 && raw[:7] == "```json" {
		// Find the closing ```
		start := 7
		for start < len(raw) && (raw[start] == '\n' || raw[start] == '\r') {
			start++
		}
		end := len(raw)
		if idx := bytes.Index([]byte(raw[start:]), []byte("```")); idx != -1 {
			end = start + idx
		}
		jsonStr = raw[start:end]
	}

	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		fmt.Println("Claude raw:", raw)
		return nil, fmt.Errorf("❌ Failed to parse Claude JSON actions")
	}

	return parsed.Actions, nil
}

/*****************************
 * Apply patch actions from AI
 *****************************/

func applyActions(actions []PatchAction) error {

	for _, a := range actions {

		switch a.Op {

		case "mkdir":
			if !isAllowedPath(a.Dir) {
				return fmt.Errorf("❌ Forbidden mkdir: %s", a.Dir)
			}
			path := filepath.Join(rootDir, a.Dir)
			fmt.Println("[mkdir]", path)
			os.MkdirAll(path, 0755)

		case "write":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("❌ Forbidden write: %s", a.File)
			}
			path := filepath.Join(rootDir, a.File)
			fmt.Println("[write]", path)
			os.MkdirAll(filepath.Dir(path), 0755)
			if err := ioutil.WriteFile(path, []byte(a.Content), 0644); err != nil {
				return err
			}

		case "delete":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("❌ Forbidden delete: %s", a.File)
			}
			path := filepath.Join(rootDir, a.File)
			fmt.Println("[delete]", path)
			os.Remove(path)
		}
	}

	return nil
}

/***************
 * Util: toJSON
 ***************/

func toJSON(v interface{}) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

/***************
 * Main Loop
 ***************/

func main() {

	for i := 1; i <= maxIters; i++ {

		fmt.Printf("\n====== ITERATION %d ======\n", i)

		files, _ := collectFiles()
		exit, testLog := runTests()

		if exit == 0 {
			fmt.Println("🎉 All tests passed! Training code is valid.")
			return
		}

		fmt.Println("❌ Tests failed, asking Claude to fix...")

		actions, err := askClaude(files, testLog)
		if err != nil {
			fmt.Println("Claude error:", err)
			return
		}

		fmt.Println("\n--- Claude Proposed Changes ---")
		for _, a := range actions {
			if a.Op == "write" {
				fmt.Println("WRITE:", a.File)
			}
			if a.Op == "mkdir" {
				fmt.Println("MKDIR:", a.Dir)
			}
			if a.Op == "delete" {
				fmt.Println("DELETE:", a.File)
			}
		}
		fmt.Println("--------------------------------")

		if err := applyActions(actions); err != nil {
			fmt.Println("Apply error:", err)
			return
		}
	}

	fmt.Println("❌ Max iterations reached without success.")
}
