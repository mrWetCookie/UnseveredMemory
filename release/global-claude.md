# Project Instructions

## CAM Protocol

**Reference**: `~/.claude/CLAUDE.md`

CAM (Continuous Architectural Memory) protocol specification, hooks, and annotation schemas are defined globally.

## Project Documentation

**Hub**: `.ai/README.md`
**CAM**: Semantically query/search the CAM database, execute scripts in the `@~/.claude/hooks` directory.

All project-specific documentation is organized in the `.ai/` directory, as well as the CAM interface database. These are the **two sources of truth** for all documentation.

### Quick Routes

| Topic | Location |
|-------|----------|
| Architecture | `.ai/core/application-architecture.md` |
| Tech Stack | `.ai/core/technology-stack.md` |
| Development | `.ai/development/development-workflow.md` |
| Testing | `.ai/development/testing-patterns.md` |
| Patterns | `.ai/patterns/` |
| Security | `.ai/patterns/security-patterns.md` |

## CAM Integration

### Automatic (Hooks)

CAM hooks fire automatically during Claude Code sessions:

- **SessionStart**: Validates CAM, loads recent context
- **UserPromptSubmit**: Queries CAM before processing your message (proactive context injection)
- **PreToolUse**: Queries CAM before tool execution
- **PostToolUse**: Annotates operations after completion
- **SessionEnd**: Summarizes session and refines knowledge graph

### Manual Commands

CAM commands are available in any initialized project via `./.claude/cam/cam.sh`:

**Core Commands:**
- **Query**: `./.claude/cam/cam.sh query "intent"`
- **Add note**: `./.claude/cam/cam.sh note "Title" "Content"`
- **Stats**: `./.claude/cam/cam.sh stats`

**Ingestion Commands (v1.5.0):**
- **Ingest file/directory**: `./.claude/cam/cam.sh ingest <path> [type]`
  - Supports: code, docs, config (auto-detected if omitted)
  - `--force` re-ingests unchanged files
  - `--dry-run` shows what would be ingested
- **Scan directory**: `./.claude/cam/cam.sh scan <directory>`
- **Check file status**: `./.claude/cam/cam.sh check-file <path>`
- **File index stats**: `./.claude/cam/cam.sh file-stats`

**Examples:**
```bash
# Ingest entire codebase
./.claude/cam/cam.sh ingest . --dry-run  # Preview first
./.claude/cam/cam.sh ingest .            # Then ingest

# Ingest specific directory with type
./.claude/cam/cam.sh ingest .ai/ docs
./.claude/cam/cam.sh ingest src/ code

# Check what needs updating
./.claude/cam/cam.sh scan .
```

### Smart CAM Triggering (Automatic)

Claude automatically consults CAM at decision points through hooks:

**SessionStart Hook (Once Per Session)**
- Validates CAM database
- Loads recent context from past work
- Queries for session patterns (what was worked on recently?)
- Cost: 1-2 seconds (one-time at session start)
- Context: ~20 tokens added to session memory

**PreToolUse Hook (Intelligent Filtering)**
- Queries CAM only at high-value decision points
- **Edit/Write operations**: Always query (code changes benefit from patterns)
- **Bash operations**: Query only if architectural (contains keywords: migrate, refactor, deploy, etc.)
- **Read/Glob/Grep**: Skip (information gathering, not decision-making)
- Results cached for 30 minutes to avoid redundant queries
- Cost per decision: ~500ms (first time), ~0ms (cached)
- Context: ~10-15 tokens per decision point

**Caching Layer**
- CAM query results cached locally for 30 minutes
- Same patterns queried multiple times return instantly from cache
- Prevents latency accumulation across session
- Old cache automatically cleaned up

**Performance Expectations**
- Session start: +1-2 seconds (one-time)
- Per code change: +500ms (first query), then cached
- Total context bloat: ~15-20 tokens per session
- Multiple edits: First triggers CAM, subsequent uses cache (instant)

**PostToolUse Auto-Ingest (v1.5.0)**
- When you Edit/Write a file, CAM automatically ingests the new content
- Smart change detection: Only re-ingests if file content actually changed
- Supports code, docs, and config files (auto-detected by extension)
- Skips large files (>100KB), lock files, and node_modules
- Keeps CAM's semantic index synchronized with codebase changes

### Proactive CAM Usage (Manual Consultation)

In addition to automatic hooks, Claude can proactively consult CAM during reasoning:

**When Making Decisions:**
- Query CAM before architectural decisions:
  ```bash
  ./.claude/cam/cam.sh query "similar architectural patterns"
  ```
- Reference historical solutions before bug fixes:
  ```bash
  ./.claude/cam/cam.sh query "similar bug fixes"
  ```
- Search for patterns before implementing features:
  ```bash
  ./.claude/cam/cam.sh query "feature implementation patterns"
  ```

**During Reasoning:**
- Explicitly cite CAM findings: "Based on CAM context, this pattern was used before..."
- Note where approach differs from CAM: "Diverging from CAM pattern because..."
- Use confidence scores: "This approach scored 0.87 in CAM retrieval"

**When Uncertain:**
- Query CAM first before deciding
- Use top results to inform implementation choices
- Document decisions that contradict CAM patterns

**Integration Points:**
- Use CAM context when refactoring existing code
- Consult CAM when modifying infrastructure (hooks, scripts, core)
- Reference CAM when adding new features to existing systems

### Cache Management

Clear CAM cache manually:
```bash
# Remove cache entries older than 30 minutes
~/.claude/hooks/cam-cache-clean.sh

# Force clear all cache
~/.claude/hooks/cam-cache-clean.sh clean-all
```

Cache is stored in: `~/.claude/.cam-cache/`
Session context cache: `~/.claude/.session-cam-context`

## Project Policies

### Changelog

Before modifying CAM infrastructure, append changes to `~/.claude/cam-template/CHANGELOG.md`:
- `~/.claude/CLAUDE.md` (protocol)
- `~/.claude/hooks/*.sh` (hooks)
- `~/.claude/cam-template/cam_core.py` / `~/.claude/cam-template/cam.sh` (core)
- `~/.claude/cam-template/settings-hooks.json` (hooks settings)

### Version Management

**When modifying `~/.claude/cam-template/`**, increment `VERSION.txt` proactively:

1. Determine change type per [VERSIONING.md](~/.claude/cam-template/VERSIONING.md):
   - Bug fix / improvement → Increment patch (e.g., 1.2.6 → 1.2.7)
   - New feature / phase → Increment minor (e.g., 1.2.9 → 1.3.1)
   - Breaking change → Increment major

2. Update these files in `~/.claude/cam-template/`:
   - `VERSION.txt` - Plain version number
   - `cam_core.py` - `CAM_VERSION` variable (line 22)
   - `CHANGELOG.md` - New `[x.y.z]` entry at top

3. After editing template, deploy changes:
   ```bash
   ~/.claude/hooks/cam-sync-template.sh
   ```

4. Projects pull updates via:
   ```bash
   ./cam.sh upgrade
   ```

### Documentation

- **Source of truth**: `.ai/` directory
- **Single location**: Each fact in ONE place
- **Cross-reference**: Never duplicate

---

*Claude reads `.ai/` for context. CAM indexes `.ai/` for semantic queries.*
