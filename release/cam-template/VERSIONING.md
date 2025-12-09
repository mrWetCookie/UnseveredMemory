# CAM Versioning Scheme

**Authority**: This versioning scheme is the source of truth for all CAM updates, migrations, and template synchronization.

---

## Version Format

```
MAJOR.MINOR.PATCH
  1    .   2    .   3
```

---

## Versioning Rules

### Patch Version Increment (x.y.z)

Increment patch version sequentially from 1 to 9:

```
1.2.1 → 1.2.2 → 1.2.3 → 1.2.4 → 1.2.5 → 1.2.6 → 1.2.7 → 1.2.8 → 1.2.9
```

**When to increment patch**:
- Bug fixes
- Hook improvements
- Performance optimizations
- Non-breaking changes to core functionality

### Minor Version Increment (x.y → x.y+1)

When patch version reaches .9, reset patch to .1 and increment minor:

```
1.2.9 → 1.3.1 (not 1.3.0)
```

Continue patching through 1.3.x:

```
1.3.1 → 1.3.2 → 1.3.3 → 1.3.4 → 1.3.5 → 1.3.6 → 1.3.7 → 1.3.8 → 1.3.9
```

Then jump to next minor:

```
1.3.9 → 1.4.1 (not 1.4.0)
```

**When to increment minor**:
- New phase completion (Phase 2.1, Phase 3, etc.)
- Major feature additions (new hooks, new query capabilities)
- Significant architecture changes

### Major Version Increment (x → x+1)

When minor reaches .9 and we're adding major breaking changes or completing major phases:

```
1.9.9 → 2.0.0 (hypothetical future)
```

**Reserved for**:
- Complete system redesigns
- Major breaking changes
- Phase transitions (Phase 1 → Phase 2, Phase 2 → Phase 3)

---

## Update Procedure

### Every time you update cam-template or hooks:

1. **Identify change type**:
   - Bug fix / improvement? → Increment patch (.z)
   - New feature / phase? → Increment minor (.y)
   - Breaking change / redesign? → Increment major (.x)

2. **Update these files in ~./claude/cam-template/**:
   - `VERSION` - Plain text file with version number
   - `cam_core.py` - Update `CAM_VERSION = "x.y.z"` (line 22)
   - `CHANGELOG.md` - Add new [x.y.z] section at top
   - Hook scripts - Update version comments if relevant

3. **Propagate to projects**:
   - Copy updated files to ~/.claude/cam-template/
   - Severance project will show "Update available" when checking `./cam.sh version`
   - Users run `./cam.sh upgrade` to get new version

4. **Document in CAM**:
   - Ingest updated CHANGELOG.md to CAM
   - Annotate version change with reason and scope
   - Add to operations.log

5. **Commit to git**:
   - Include version number in commit message
   - Document changes in CHANGELOG
   - Example: `chore: Bump to 1.2.5 - hook optimization`

---

## Current Versioning Timeline

```
1.2.1 - Initial release with core + basic hooks
1.2.2 - PostToolUse race condition fix (cam-note.sh guard)
1.2.3 - Expanded annotation scope (all operations, not just "significant")
1.2.4 - [Next patch increment]
...
1.2.9 - [Last patch in 1.2 cycle]
1.3.1 - Phase 2.1 begins: Bootstrap + Batch Refinement
1.3.2 - Bootstrap procedure implemented
1.3.3 - Periodic batch refinement implemented
1.3.4 - Success rate tracking implemented
...
1.3.9 - [Last patch in 1.3 cycle]
1.4.1 - Phase 3 begins: Cross-project learning
...
2.0.0 - [Future major redesign, if needed]
```

---

## Files That Track Version

**All must be kept in sync**:

| File | Location | Format | Purpose |
|------|----------|--------|---------|
| `VERSION` | `~/.claude/cam-template/VERSION` | Plain text | `1.2.3` |
| `cam_core.py` | `~/.claude/cam-template/cam_core.py` | Python var | `CAM_VERSION = "1.2.3"` |
| `CHANGELOG.md` | `~/.claude/cam-template/CHANGELOG.md` | Markdown | `## [1.2.3] - YYYY-MM-DD` |
| `CLAUDE.md` | `~/.claude/CLAUDE.md` (optional) | Markdown | Reference if versioning mentioned |
| Project `VERSION` | `<project>/.claude/cam/VERSION` | Plain text | Syncs from template |
| Project `cam_core.py` | `<project>/.claude/cam/cam_core.py` | Python var | Syncs from template |

---

## Verification Checklist

Before committing a version bump:

- [ ] VERSION file updated in template
- [ ] cam_core.py CAM_VERSION updated in template
- [ ] CHANGELOG.md has new [x.y.z] entry at top
- [ ] All hook scripts have consistent version references
- [ ] Severance project synced to new version (if applicable)
- [ ] CAM annotated with version change
- [ ] Git commit includes version number in message
- [ ] Template and projects both show same version in `./cam.sh version`

---

## Rationale

**Why x.y.1 → x.y.9 instead of x.y.0 → x.y.9?**

- Avoids .0 releases (less clear semantically)
- Aligns with "9 phases per minor version" philosophy
- Cleaner visual progression: 1.2.1, 1.2.2, ... 1.2.9, 1.3.1

**Why jump from x.y.9 to (x).(y+1).1?**

- Maintains consistency (always .1 to .9 per minor version)
- Clear boundary markers (any .9 means "last of this cycle")
- Reserved .0 for future major version changes only

---

## Implementation Notes

- This scheme is now documented in CAM (ingested as knowledge)
- All future updates must reference this VERSIONING.md
- Any deviation should be explicitly documented and annotated to CAM
- Projects can reference this to understand version compatibility
