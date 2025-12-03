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
	"regexp"
	"strings"
)

const (
	rootDir  = "project"
	maxIters = 10

	// Your customized API settings
	anthropicURL = "https://api.deerapi.com/v1/messages"
	modelName    = "claude-sonnet-4-5-20250929"
	key          = ""
	// anthropicURL = "https://api.siray.ai/v1/messages"
	// modelName    = "anthropic/claude-sonnet-4.5"
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

type PaperAnalysisResult struct {
	Features string `json:"features"`
	Label    string `json:"label"`
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
			continue // never read subdirectories
		}

		// 🚫 跳过 config_default.json
		if entry.Name() == "config_default.json" {
			continue
		}

		filePath := filepath.Join(rootDir, entry.Name())
		content, err := ioutil.ReadFile(filePath)
		if err != nil {
			return nil, err
		}

		files[entry.Name()] = string(content)
	}

	return files, nil
}

// ----- RUN MAIN.PY --FAST -----

func runFast() (int, string) {
	cmd := exec.Command("project/venv/bin/python3", "project/main.py", "--test")
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

	output := out.String()

	// Print output to console
	fmt.Println("\n===== MAIN.PY OUTPUT =====")
	fmt.Println(output)
	fmt.Println("===========================\n")

	return exitCode, output
}

// ----- JSON EXTRACTION -----

// extractLongestJSON uses regex to find all potential JSON objects/arrays
// in the text, validates them, and returns the longest valid JSON string.
func extractLongestJSON(text string) (string, error) {
	// Regex patterns to find JSON objects and arrays
	// We look for balanced braces/brackets with any content
	patterns := []*regexp.Regexp{
		// Match JSON objects: { ... }
		regexp.MustCompile(`\{(?:[^{}]|\{[^{}]*\})*\}`),
		// Match JSON arrays: [ ... ]
		regexp.MustCompile(`\[(?:[^\[\]]|\[[^\[\]]*\])*\]`),
	}

	var candidates []string

	// Find all potential JSON strings
	for _, pattern := range patterns {
		matches := pattern.FindAllString(text, -1)
		candidates = append(candidates, matches...)
	}

	if len(candidates) == 0 {
		return "", fmt.Errorf("no JSON-like structures found")
	}

	// Try to parse each candidate and keep track of the longest valid one
	var longestValid string
	longestLen := 0

	for _, candidate := range candidates {
		// Try to unmarshal to verify it's valid JSON
		var testObj interface{}
		if err := json.Unmarshal([]byte(candidate), &testObj); err == nil {
			// Valid JSON - check if it's the longest
			serialized, _ := json.Marshal(testObj)
			if len(serialized) > longestLen {
				longestLen = len(serialized)
				longestValid = candidate
			}
		}
	}

	if longestValid == "" {
		return "", fmt.Errorf("no valid JSON found in text")
	}

	return strings.TrimSpace(longestValid), nil
}

// ----- CLAUDE CALL -----

func askClaude(projectFiles map[string]string, trainOutput string) ([]PatchAction, error) {

	// ----- READ system_prompt.txt -----
	promptBytes, err := ioutil.ReadFile("system_prompt.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read system_prompt.txt: %v", err)
	}
	systemPrompt := string(promptBytes)

	// ----- BUILD REQUEST BODY -----
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": 60000,
		"system":     systemPrompt,
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
	// req.Header.Set("Authorization", "Bearer "+os.Getenv("ANTHROPIC_AUTH_TOKEN"))
	req.Header.Set("Authorization", "Bearer "+key)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, _ := ioutil.ReadAll(resp.Body)

	var claude ClaudeResponse
	if err := json.Unmarshal(respBytes, &claude); err != nil {
		fmt.Println("Raw:", string(respBytes))
		return nil, err
	}

	if len(claude.Content) == 0 {
		fmt.Println("Raw:", string(respBytes))
		return nil, fmt.Errorf("Claude response has no content")
	}

	raw := claude.Content[0].Text

	// ========= EXTRACT LONGEST VALID JSON =========
	clean, err := extractLongestJSON(raw)
	if err != nil {
		fmt.Println("Claude raw output:\n", raw)
		return nil, fmt.Errorf("failed to extract valid JSON: %v", err)
	}

	// Try to parse as object with "actions" field first
	var parsed struct {
		Actions []PatchAction `json:"actions"`
	}
	var actions []PatchAction

	if err := json.Unmarshal([]byte(clean), &parsed); err == nil && len(parsed.Actions) > 0 {
		// Successfully parsed as {"actions": [...]}
		actions = parsed.Actions
	} else {
		// Try parsing as plain array [...]
		if err := json.Unmarshal([]byte(clean), &actions); err != nil {
			fmt.Println("Claude raw output:\n", raw)
			fmt.Println("Extracted JSON:\n", clean)
			fmt.Print(err)
			return nil, fmt.Errorf("Claude did not return valid JSON actions")
		}
	}

	// ----- PRINT TO STDOUT -----
	pretty, _ := json.MarshalIndent(actions, "", "  ")
	fmt.Println("\n===== CLAUDE PATCH (stdout) =====")
	fmt.Println(string(pretty))
	fmt.Println("=================================\n")

	return actions, nil
}

// ----- PAPER ANALYSIS -----

func analyzePaperWithClaude() (*PaperAnalysisResult, error) {
	// Read paper.md
	paperPath := filepath.Join(rootDir, "paper.md")
	paperContent, err := ioutil.ReadFile(paperPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read paper.md: %v", err)
	}

	promptBytes, err := ioutil.ReadFile("feature_prompt.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read feature_prompt.txt: %v", err)
	}
	featurePrompt := string(promptBytes)
	prompt := featurePrompt + string(paperContent)

	// Build request body
	body := map[string]interface{}{
		"model":      modelName,
		"max_tokens": 4000,
		"system":     "You are a helpful assistant that analyzes research papers and extracts feature and label specifications for time series forecasting projects. Always respond with valid JSON only.",
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]string{
					{"type": "text", "text": prompt},
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

	var claude ClaudeResponse
	if err := json.Unmarshal(respBytes, &claude); err != nil {
		fmt.Println("Raw response:", string(respBytes))
		return nil, err
	}

	if len(claude.Content) == 0 {
		fmt.Println("Raw response:", string(respBytes))
		return nil, fmt.Errorf("Claude response has no content")
	}

	raw := claude.Content[0].Text

	// Extract longest valid JSON
	clean, err := extractLongestJSON(raw)
	if err != nil {
		fmt.Println("Claude raw output:\n", raw)
		return nil, fmt.Errorf("failed to extract valid JSON: %v", err)
	}

	// First, check if clean is a general JSON object
	var testObj interface{}
	if err := json.Unmarshal([]byte(clean), &testObj); err != nil {
		fmt.Println("Claude raw output:\n", raw)
		fmt.Println("Extracted JSON:\n", clean)
		return nil, fmt.Errorf("Claude did not return valid JSON for paper analysis")
	}

	// If it's a JSON object (map), write it directly to config_paper.json
	if _, isMap := testObj.(map[string]interface{}); isMap {
		configPath := filepath.Join(rootDir, "config_paper.json")

		// Pretty print the JSON
		prettyJSON, err := json.MarshalIndent(testObj, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("failed to marshal JSON: %v", err)
		}

		if err := ioutil.WriteFile(configPath, prettyJSON, 0644); err != nil {
			return nil, fmt.Errorf("failed to write config_paper.json: %v", err)
		}

		fmt.Println("\n===== GENERATED config_paper.json =====")
		fmt.Println(string(prettyJSON))
		fmt.Println("========================================\n")

		// Return a result indicating we wrote the config directly
		return &PaperAnalysisResult{
			Features: "written to config_paper.json",
			Label:    "written to config_paper.json",
		}, nil
	}

	// Otherwise, try to parse as PaperAnalysisResult
	var result PaperAnalysisResult
	if err := json.Unmarshal([]byte(clean), &result); err != nil {
		fmt.Println("Claude raw output:\n", raw)
		fmt.Println("Extracted JSON:\n", clean)
		return nil, fmt.Errorf("Claude did not return valid JSON for paper analysis")
	}

	// Default to Alpha158 and standard label if empty
	if result.Features == "" {
		result.Features = "Alpha158"
	}
	if result.Label == "" {
		result.Label = "Ref($close, -1)/$close-1"
	}

	return &result, nil
}

func generateConfigPaper(analysis *PaperAnalysisResult) error {
	configPath := filepath.Join(rootDir, "config_paper.json")

	// Create config_paper.json structure
	config := map[string]interface{}{
		"model": map[string]interface{}{},
		"train": map[string]interface{}{},
		"data": map[string]interface{}{
			"features": analysis.Features,
			"label":    analysis.Label,
		},
	}

	// If features is a JSON array string, parse it
	if strings.HasPrefix(analysis.Features, "[") {
		var featureList interface{}
		if err := json.Unmarshal([]byte(analysis.Features), &featureList); err == nil {
			config["data"].(map[string]interface{})["features"] = featureList
		}
	}

	jsonBytes, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(configPath, jsonBytes, 0644); err != nil {
		return err
	}

	fmt.Println("\n===== GENERATED config_paper.json =====")
	fmt.Println(string(jsonBytes))
	fmt.Println("========================================\n")

	return nil
}

// ----- ALLOWED PATHS -----

func isAllowedPath(rel string) bool {
	return rel == "model.py" ||
		rel == "config_paper.json" ||
		rel == "requirements.txt" // NOW ALLOWED
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

	// ----- AUTO INSTALL DEPENDENCIES -----
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
	// ----- STEP 0: ANALYZE PAPER AND GENERATE CONFIG -----
	fmt.Println("\n====== ANALYZING PAPER.MD ======")
	analysis, err := analyzePaperWithClaude()
	if err != nil {
		fmt.Println("⚠️  Warning: Could not analyze paper.md:", err)
		fmt.Println("⚠️  Continuing with default config_paper.json (empty)")
	} else {
		// If analysis was written directly to config_paper.json, skip generateConfigPaper
		if analysis.Features == "written to config_paper.json" && analysis.Label == "written to config_paper.json" {
			fmt.Println("✓ config_paper.json written directly from Claude response")
		} else {
			fmt.Printf("✓ Extracted features: %v\n", analysis.Features)
			fmt.Printf("✓ Extracted label: %s\n", analysis.Label)

			if err := generateConfigPaper(analysis); err != nil {
				fmt.Println("⚠️  Warning: Could not generate config_paper.json:", err)
			} else {
				fmt.Println("✓ Generated config_paper.json")
			}
		}
	}
	fmt.Println("================================\n")

	// ----- MAIN TRAINING LOOP -----
	var lastTrainOutput string = "Initial generation - no previous test output."

	for iter := 1; iter <= maxIters; iter++ {
		fmt.Printf("\n====== ITERATION %d ======\n", iter)

		// Step 1: Collect current files
		files, _ := collectFiles()

		// Step 2: Ask Claude to generate/fix code
		fmt.Println("🤖 Asking Claude to generate/fix code...")
		actions, err := askClaude(files, lastTrainOutput)
		if err != nil {
			fmt.Println("Claude error:", err)
			return
		}

		// Step 3: Apply Claude's changes
		if err := applyActions(actions); err != nil {
			fmt.Println("Apply error:", err)
			return
		}

		// Step 4: Run the test
		fmt.Println("🧪 Running main.py --test...")
		exit, output := runFast()
		lastTrainOutput = output

		// Step 5: Check if test passed
		if exit == 0 {
			fmt.Println("🎉 SUCCESS! main.py --test completed successfully.")
			return
		}

		fmt.Println("❌ Test failed. Will retry in next iteration...")
	}

	fmt.Println("❌ Max iterations reached without success.")
}
