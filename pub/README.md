# Publication Draft

This directory contains the draft of the scientific publication about batch effects in Raman spectroscopy.

## Files

- `main-text.md` - Main publication text
- `figures/` - Publication figures

## Linting and Formatting

This directory includes its own Node.js tooling for markdown linting, spell checking, and prose linting.

### Setup

```bash
npm install
```

### Commands

```bash
# Run all checks
npm run lint:all

# Run just markdown and spelling
npm run lint

# Individual checks
npm run lint:md      # Markdown structure
npm run lint:spell   # Spell checking
npm run lint:links   # Check for broken links
npm run lint:prose   # Prose style (requires vale CLI)

# Auto-fix markdown issues
npm run format
```

## Tools

- **markdownlint-cli2**: Enforces markdown structure and style
- **cspell**: Spell checking with scientific terminology dictionary
- **markdown-link-check**: Validates links
- **vale**: Prose linting (optional, requires separate installation)
