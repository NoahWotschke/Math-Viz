# Development Setup Guide

## Quick Start

### 1. Set up the virtual environment
```bash
cd math-viz
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r mathvis-core/requirements.txt
pip install -e mathvis-core/
```

### 3. Install the CLI or Web UI
```bash
# For CLI development
pip install -e mathvis-cli/

# For Web UI development
pip install -e mathvis-web/
```

## VS Code Configuration

The workspace is configured with:
- `pyrightconfig.json` - Tells Pylance where to find PDEs modules
- `.vscode/settings.json` - Points to the virtual environment

**If you see import errors in Pylance:**
1. Make sure `.venv` is activated: `source .venv/bin/activate`
2. Reload VS Code: `Cmd+Shift+P` → `Developer: Reload Window`
3. Clear Pylance cache: `Cmd+Shift+P` → `Pylance: Clear Cache`

## Running the CLI

```bash
source .venv/bin/activate
python3 mathvis-cli/solve.py --help
```

## Running the Web UI

```bash
source .venv/bin/activate
cd mathvis-web
streamlit run app.py
```

## Troubleshooting

### "Import could not be resolved" errors
- Make sure you've run `pip install -e mathvis-core/`
- Reload VS Code window
- Check that `.venv/bin/python` is the selected interpreter

### "streamlit not found"
- Install with: `pip install streamlit`

### "numpy not found"
- Install dependencies: `pip install -r mathvis-core/requirements.txt`
