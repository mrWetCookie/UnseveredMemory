# CAM Commands

Complete command reference for CAM (Continuous Architectural Memory).

---

## Core Commands

```bash
# Show CAM version
./cam.sh version
```

```bash
# Show CAM statistics (embedding count, database sizes, graph metrics)
./cam.sh stats
```

```bash
# Upgrade CAM from global template (~/.claude/cam-template/)
./cam.sh upgrade

# Force upgrade even if versions match
./cam.sh upgrade --force
```

---

## Semantic Search

```bash
# Query CAM with semantic search
# Returns top 5 results by default, ranked by cosine similarity
./cam.sh query "<search_text>"

# Parameters:
#   <search_text>  - Natural language query describing your intent
#   [top_k]        - Optional: Number of results to return (default: 5)

# Examples:
./cam.sh query "authentication patterns"
./cam.sh query "how does the API handle errors" 10
./cam.sh query "database schema" --type docs
```

```bash
# Retrieve full embedding content by ID
# Use after query to get complete content of a result
./cam.sh get <embedding_id>

# Parameters:
#   <embedding_id>  - 16-character hex ID from query results

# Example:
./cam.sh get 572ff1840d0f0a99
```

---

## Notes & Annotations

```bash
# Add an ephemeral note to CAM
# Notes are embedded and searchable like any other content
./cam.sh note "<title>" "<content>" [tags]

# Parameters:
#   <title>    - Short descriptive title
#   <content>  - Note content (can be multi-line)
#   [tags]     - Optional: Comma-separated tags for categorization

# Examples:
./cam.sh note "Bug Fix" "Fixed authentication timeout issue"
./cam.sh note "Architecture Decision" "Chose PostgreSQL over MongoDB" "decision,database"
./cam.sh note "TODO" "Refactor user service" "todo,refactor,user"
```

```bash
# Add manual annotation to an existing embedding
./cam.sh annotate "<content>"

# Parameters:
#   <content>  - Annotation text to add
```

---

## File Ingestion

```bash
# Ingest file or directory into CAM
# Automatically detects file type and creates embeddings
./cam.sh ingest <path> [type] [flags]

# Parameters:
#   <path>      - File or directory path to ingest
#   [type]      - Optional: Force type (code, docs, config)
#   [--force]   - Re-ingest even if file unchanged
#   [--dry-run] - Preview what would be ingested without doing it

# Examples:
./cam.sh ingest src/                     # Ingest source directory (auto-detect)
./cam.sh ingest .ai/ docs                # Ingest as documentation
./cam.sh ingest config.yaml config       # Ingest as config
./cam.sh ingest . --dry-run              # Preview full project ingestion
./cam.sh ingest src/ --force             # Force re-ingest all files
```

```bash
# Scan directory for new or modified files
# Reports status without ingesting
./cam.sh scan <directory>

# Parameters:
#   <directory>  - Directory path to scan

# Example:
./cam.sh scan .
./cam.sh scan src/
```

```bash
# Check single file status in CAM index
# Shows if file is new, modified, or unchanged
./cam.sh check-file <path>

# Parameters:
#   <path>  - File path to check

# Example:
./cam.sh check-file src/main.py
```

```bash
# Show file index statistics
# Displays counts by type, total size, last ingestion times
./cam.sh file-stats
```

---

## Knowledge Graph

```bash
# Build knowledge graph from embeddings
# Discovers temporal, semantic, and causal relationships
./cam.sh graph build

# Rebuild graph from scratch (clears existing)
./cam.sh graph build --rebuild
```

```bash
# Show graph statistics
# Displays node count, edge count, relationship types
./cam.sh graph stats
```

```bash
# Create manual relationship between embeddings
./cam.sh relate <source_id> <target_id> <type> [weight]

# Parameters:
#   <source_id>   - Source embedding ID
#   <target_id>   - Target embedding ID
#   <type>        - Relationship type (temporal, semantic, causal, modifies, references)
#   [weight]      - Optional: Relationship strength 0.0-1.0 (default: 1.0)

# Example:
./cam.sh relate abc123 def456 references 0.8
```

```bash
# Find embedding ID by source file path
# Useful for creating relationships to documents
./cam.sh find-doc <file_path>

# Parameters:
#   <file_path>  - Path to source file

# Example:
./cam.sh find-doc ".ai/patterns/api-routing.md"
```

---

## Session Management

```bash
# Show active session primers
# Displays primers cached for post-compact recovery
./cam.sh primer-status
```

```bash
# List recent sessions with metadata
./cam.sh sessions [--limit N]

# Parameters:
#   [--limit N]  - Optional: Number of sessions to show (default: 10)
```

```bash
# Get most recent session summary
./cam.sh last-session
```

```bash
# Get specific session details by ID
./cam.sh session <session_id>

# Parameters:
#   <session_id>  - Full or partial session ID
```

---

## Ralph Wiggum Integration

```bash
# List recent Ralph loops
./cam.sh ralph-loops [flags]

# Flags:
#   [--limit N]       - Number of loops to show (default: 10)
#   [--successful]    - Only show successful loops
#   [--outcome TYPE]  - Filter by outcome (success, max_iterations, cancelled)

# Examples:
./cam.sh ralph-loops
./cam.sh ralph-loops --successful
./cam.sh ralph-loops --limit 5 --outcome max_iterations
```

```bash
# View iteration history for a specific loop
./cam.sh ralph-history <loop_id>

# Parameters:
#   <loop_id>  - Ralph loop ID from ralph-loops output
```

```bash
# Find patterns from similar past tasks
# Searches CAM for loops with similar prompts
./cam.sh ralph-patterns "<task_description>"

# Parameters:
#   <task_description>  - Description of current task

# Example:
./cam.sh ralph-patterns "Build REST API for user management"
```

---

## Evaluation & Benchmarks

```bash
# Run evaluation suite
./cam.sh eval <type>

# Types:
#   embeddings  - STS correlation test (Pearson/Spearman)
#   retrieval   - Precision/Recall/nDCG metrics
#   graph       - Edge coherence, temporal consistency
#   extrinsic   - Operational utility analysis
#   all         - Run all intrinsic evaluations

# Example:
./cam.sh eval all
```

```bash
# Run benchmarks
./cam.sh benchmark <type>

# Types:
#   dmr    - Deep Memory Retrieval benchmark
#   locomo - Long-Context Memory benchmark
#   all    - Run all benchmarks
```

---

## Troubleshooting

```bash
# Verify hooks are registered in settings
cat ~/.claude/settings.json | jq '.hooks | keys'
```

```bash
# Check specific hook configuration
cat ~/.claude/settings.json | jq '.hooks.SessionStart'
```

```bash
# Verify Gemini API key is configured
cat ~/.claude/hooks/.env | grep GEMINI_API_KEY
```

```bash
# Test Gemini API connection
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$(grep GEMINI_API_KEY ~/.claude/hooks/.env | cut -d= -f2)"
```

```bash
# Initialize CAM in current project
~/.claude/hooks/init-cam.sh
```

```bash
# Verify CAM installation
ls -la .claude/cam/
```

```bash
# Check CAM version matches template
./cam.sh version
cat ~/.claude/cam-template/VERSION.txt
```

---
