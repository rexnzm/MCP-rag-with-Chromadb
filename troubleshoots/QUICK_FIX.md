# Quick Fix: Ollama Connection Closed Error

## Immediate Steps to Resolve "Connection Forcibly Closed"

### Step 1: Run Diagnostic Tool

```bash
cd D:\Documents\ml_python_courses\MCP\mcp_rag_chroma
python check_ollama.py
```

This will tell you exactly what's wrong.

---

### Step 2: Most Likely Fixes

#### Fix #1: Restart Ollama (Most Common)

**Windows:**
1. Look for Ollama icon in system tray (bottom-right corner)
2. Right-click → Quit Ollama
3. Start Ollama again from Start Menu
4. Wait 10 seconds for it to fully start
5. Run `python check_ollama.py` again

**Mac:**
```bash
pkill ollama
ollama serve
```

**Linux:**
```bash
sudo systemctl restart ollama
```

---

#### Fix #2: Install/Reinstall the Model

```bash
# Remove old model (if exists)
ollama rm nomic-embed-text

# Pull fresh copy
ollama pull nomic-embed-text

# Verify
ollama list
```

You should see:
```
NAME                    ID              SIZE
nomic-embed-text:latest abc123def456    274 MB
```

---

#### Fix #3: Check Available Memory

The embedding model needs ~4GB RAM. If your system has less than 8GB total, Ollama may crash.

**Windows - Check Memory:**
1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to Performance tab
3. Check available RAM

**If low on memory:**
1. Close unnecessary applications (browsers, IDEs, etc.)
2. Restart your computer
3. Start Ollama before other applications

---

### Step 3: Verify the Fix

```bash
# Test Ollama directly
curl http://localhost:11434/api/tags

# Run full diagnostic
python check_ollama.py

# Test from server (if check_ollama.py passes)
python server.py
```

You should see:
```
✓ Ollama connection validated successfully
```

---

## Alternative: Use Without Ollama (Temporary)

If Ollama keeps crashing, you can temporarily use a different embedding service.

Edit `server.py` and replace the embeddings initialization:

```python
# Comment out Ollama embeddings
# embeddings = OllamaEmbeddings(
#     model=EMBED_MODEL,
#     base_url=OLLAMA_BASE_URL
# )

# Use HuggingFace embeddings instead (requires: pip install sentence-transformers)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Smaller, faster model
)
```

**Note:** This requires installing additional packages:
```bash
pip install sentence-transformers
```

---

## Still Not Working?

### Check Ollama Installation

```bash
# Verify Ollama is installed
ollama --version

# If not found, reinstall from:
# https://ollama.ai/download
```

### Check Port Conflicts

```bash
# Windows
netstat -ano | findstr :11434

# Mac/Linux
lsof -i :11434
```

If something else is using port 11434, kill it or configure Ollama to use a different port.

### Check Firewall

Temporarily disable Windows Defender Firewall or antivirus to test if they're blocking the connection.

---

## Expected Output After Fix

When everything is working, `python check_ollama.py` should show:

```
==================================================
1. Checking if Ollama service is running...
==================================================
✓ Ollama service is running

==================================================
2. Checking if embedding model is installed...
==================================================

Installed models (1):
  - nomic-embed-text:latest (274.0 MB)

✓ Model 'nomic-embed-text:latest' is installed

==================================================
3. Testing embedding generation...
==================================================
Requesting embedding for: 'This is a test sentence.'
Using model: nomic-embed-text
✓ Successfully generated embedding
  Embedding dimension: 768
  Sample values: [0.123, -0.456, 0.789, ...]...

==================================================
DIAGNOSTIC SUMMARY
==================================================
SERVICE: ✓ PASS
MODEL: ✓ PASS
EMBEDDING: ✓ PASS

✓ All checks passed! Ollama is properly configured.
```

---

## Contact for Help

If none of these fixes work, gather this information:

1. Output of `python check_ollama.py`
2. Output of `ollama list`
3. Your OS and RAM amount
4. Full error message from server.py

Then consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for advanced solutions.
