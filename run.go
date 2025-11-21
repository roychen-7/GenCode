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
	rootDir  = "project"
	maxIters = 10

	// Your Siray.ai API endpoint + model
	anthropicURL = "https://api.siray.ai/v1/messages"
	modelName    = "anthropic/claude-sonnet-4.5"
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
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": 6000,
		"system": `
You are an autonomous coding agent.

You are ONLY allowed to modify these exact files in the project root:

1) model.py
2) config.py
3) train.py

You MUST use project/paper.md as the source of truth for model architecture and training logic.
paper.md is READ-ONLY. NEVER modify it.

Forbidden:
- paper.md
- data/**
- datasets.py
- run.go
- any python files other than model.py/config.py/train.py
- any directories such as models/, model/, layers/, utils/

Your output MUST be FULL JSON:
{
  "actions": [
    { "op": "write", "file": "model.py", "content": "FULL file content" }
  ]
}

Rules:
- Only full-file writes.
- No diffs.
- No explanations outside JSON.
`,
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
	req.Header.Set("Authorization", "Bearer "+os.Getenv("ANTHROPIC_AUTH_TOKEN"))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, _ := ioutil.ReadAll(resp.Body)

	// Parse Claude response
	var claude ClaudeResponse
	if err := json.Unmarshal(respBytes, &claude); err != nil {
		fmt.Println("Raw:", string(respBytes))
		return nil, err
	}

	raw := claude.Content[0].Text

	// Parse JSON actions
	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}

	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		fmt.Println("Claude raw output:\n", raw)
		return nil, fmt.Errorf("Claude did not return valid JSON actions")
	}

	// ----- B: PRINT PATCH TO STDOUT -----
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
	for _, a := range acts {

		switch a.Op {

		case "mkdir":
			// Absolutely forbidden — no directories allowed
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

		case "delete":
			// We forbid deleting any file (even allowed ones)
			return fmt.Errorf("Forbidden delete: %s", a.File)
		}
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
