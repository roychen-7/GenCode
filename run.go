package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
)

const (
	rootDir  = "project"
	maxIters = 10

	// Your Siray.ai API endpoint + model
	// anthropicURL = "https://api.siray.ai/v1/messages"
	anthropicURL = "https://api.deerapi.com/v1/messages"
	// modelName    = "anthropic/claude-sonnet-4.5"
	modelName = "claude-sonnet-4-5-20250929"
	key       = "sk-JTWR4AtPEeITsqtM8EZ4DVrPc7QRRY31ESlelVAAtkOwo0uA"
)

// ----- DATA STRUCTURES -----

type PatchAction struct {
	Op      string `json:"op"`
	File    string `json:"file,omitempty"`
	Dir     string `json:"dir,omitempty"`
	Content string `json:"content,omitempty"`
}

type ClaudeResponse struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

// ----- FILE COLLECTION -----

func collectFiles() (map[string]string, error) {
	files := make(map[string]string)

	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			// skip ALL folders (example: data/, cn_data/, anything/)
			continue
		}

		// Only process files directly under project/
		filePath := filepath.Join(rootDir, entry.Name())
		content, err := ioutil.ReadFile(filePath)
		if err != nil {
			return nil, err
		}

		files[entry.Name()] = string(content)
	}

	return files, nil
}

// ----- RUN TRAIN.PY --FAST -----

func runFastTrain() (int, string) {
	cmd := exec.Command("python3", "project/train.py", "--fast")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
		}
	}

	return exitCode, out.String()
}

// ----- CLAUDE CALL -----

func askClaude(projectFiles map[string]string, trainOutput string) ([]PatchAction, error) {

	// ----- READ SYSTEM PROMPT FROM system_prompt.txt -----
	promptBytes, err := ioutil.ReadFile("system_prompt.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read system_prompt.txt: %v", err)
	}
	systemPrompt := string(promptBytes)

	// ----- BUILD REQUEST BODY -----
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": 60000,
		"system":     systemPrompt, // ← USE EXTERNAL PROMPT
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]string{
					{"type": "text", "text": "PROJECT FILES:\n" + toJSON(projectFiles)},
					{"type": "text", "text": "TRAIN OUTPUT:\n" + trainOutput},
					{"type": "text", "text": "Generate JSON patch actions."},
				},
			},
		},
	}

	jsonBytes, _ := json.Marshal(body)

	req, _ := http.NewRequest("POST", anthropicURL, bytes.NewBuffer(jsonBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, _ := ioutil.ReadAll(resp.Body)

	fmt.Println("1")

	var claude ClaudeResponse
	if err := json.Unmarshal(respBytes, &claude); err != nil {
		fmt.Println("Raw:", string(respBytes))
		return nil, err
	}
	fmt.Println("2")

	if len(claude.Content) == 0 {
		fmt.Println("Raw:", string(respBytes))
		return nil, fmt.Errorf("Claude response has no content")
	}
	fmt.Println("3")

	raw := claude.Content[0].Text

	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}

	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		fmt.Println("Claude raw output:\n", raw)
		return nil, fmt.Errorf("Claude did not return valid JSON actions")
	}

	// ----- PRINT TO STDOUT -----
	pretty, _ := json.MarshalIndent(parsed.Actions, "", "  ")
	fmt.Println("\n===== CLAUDE PATCH (stdout) =====")
	fmt.Println(string(pretty))
	fmt.Println("=================================\n")

	return parsed.Actions, nil
}

// ----- SECURITY: ONLY 3 FILES ALLOWED -----

func isAllowedPath(rel string) bool {
	return rel == "model.py" || rel == "config.py" || rel == "train.py"
}

// ----- APPLY PATCHES -----

func applyActions(acts []PatchAction) error {
	reqUpdated := false

	for _, a := range acts {
		switch a.Op {

		case "mkdir":
			return fmt.Errorf("Forbidden mkdir: %s", a.Dir)

		case "write":
			if !isAllowedPath(a.File) {
				return fmt.Errorf("Forbidden write: %s", a.File)
			}

			path := filepath.Join(rootDir, a.File)
			fmt.Println("[write]", path)

			if err := ioutil.WriteFile(path, []byte(a.Content), 0644); err != nil {
				return err
			}

			if a.File == "requirements.txt" {
				reqUpdated = true
			}

		case "delete":
			return fmt.Errorf("Forbidden delete: %s", a.File)
		}
	}

	// ----- 自动安装依赖 -----
	if reqUpdated {
		fmt.Println("📦 requirements.txt updated. Installing dependencies...")

		cmd := exec.Command("pip3", "install", "-r", "project/requirements.txt")
		var out bytes.Buffer
		cmd.Stdout = &out
		cmd.Stderr = &out

		err := cmd.Run()
		fmt.Println(out.String())

		if err != nil {
			return fmt.Errorf("pip install failed: %v", err)
		}

		fmt.Println("📦 Dependencies installed successfully.")
	}

	return nil
}

// ----- JSON FORMATTER -----

func toJSON(v interface{}) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

// ----- MAIN LOOP -----

func main() {

	for iter := 1; iter <= maxIters; iter++ {
		fmt.Printf("\n====== ITERATION %d ======\n", iter)

		files, _ := collectFiles()
		exit, output := runFastTrain()

		if exit == 0 {
			fmt.Println("🎉 SUCCESS! train.py --fast completed successfully.")
			return
		}

		fmt.Println("❌ fast train failed, asking Claude to fix...")

		actions, err := askClaude(files, output)
		if err != nil {
			fmt.Println("Claude error:", err)
			return
		}

		if err := applyActions(actions); err != nil {
			fmt.Println("Apply error:", err)
			return
		}
	}

	fmt.Println("❌ Max iterations reached without success.")
}
