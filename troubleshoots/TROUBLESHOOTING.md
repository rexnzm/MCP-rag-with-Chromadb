# Troubleshooting Guide

## Ollama Connection Issues

### Error: "Connection forcibly closed" or "ConnectionError"

This error occurs when the Ollama service crashes or becomes unresponsive during an operation.

#### Quick Diagnostic

Run the diagnostic tool:
```bash
python check_ollama.py
```

This will check:
1. Is Ollama service running?
2. Is the embedding model installed?
3. Can embeddings be generated?
4. System resources (RAM/CPU)

#### Common Causes and Solutions

**1. Ollama Service Not Running**

**Symptoms:**
- Cannot connect to Ollama service
- Connection refused errors

**Solution:**
```bash
# Windows: Check system tray for Ollama icon
# If not running, start from Start Menu

# Mac/Linux: Start Ollama
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

---

**2. Embedding Model Not Installed**

**Symptoms:**
- Model not found errors
- 404 responses from Ollama

**Solution:**
```bash
# Pull the embedding model
ollama pull nomic-embed-text

# Verify it's installed
ollama list
```

You should see `nomic-embed-text:latest` in the list.

---

**3. Ollama Running Out of Memory**

**Symptoms:**
- Connection closes during embedding generation
- Ollama becomes unresponsive
- Computer slows down significantly

**Solution:**

Check available RAM:
```bash
# The check_ollama.py script shows this
python check_ollama.py
```

The `nomic-embed-text` model requires approximately **4GB of RAM**. If you have less than 8GB total RAM, Ollama may struggle.

**Options:**
- Close other applications to free memory
- Restart Ollama to clear its memory cache
- Consider using a smaller embedding model
- Upgrade your RAM (if feasible)

---

**4. Port Already in Use**

**Symptoms:**
- Ollama won't start
- Address already in use errors

**Solution:**

Check if port 11434 is in use:
```bash
# Windows
netstat -ano | findstr :11434

# Mac/Linux
lsof -i :11434
```

Kill the conflicting process or configure Ollama to use a different port.

---

**5. Ollama Process Crashed**

**Symptoms:**
- Connection worked, then suddenly failed
- Cannot reconnect to Ollama
- Empty or incomplete responses

**Solution:**

**Windows:**
1. Open Task Manager
2. Look for "ollama" processes
3. End any hung processes
4. Restart Ollama from Start Menu

**Mac/Linux:**
```bash
# Find and kill Ollama
pkill ollama

# Restart
ollama serve
```

---

**6. Firewall or Antivirus Blocking**

**Symptoms:**
- Connection works sometimes but not others
- Timeouts or connection refused

**Solution:**

Add exception for Ollama in:
- Windows Defender Firewall
- Third-party antivirus software
- Corporate firewall settings

Allow connections to `localhost:11434`.

---

### Using MCP Tools to Diagnose

The server includes a diagnostic tool:

```python
# From Claude Desktop or MCP client
check_ollama()
```

This returns:
- Service status
- Model availability
- Embedding test results
- Specific error messages
- Suggested solutions

---

## Installation Issues

### Missing Python Dependencies

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install PyPDF2 python-docx beautifulsoup4 python-pptx openpyxl odfpy
```

---

### Incompatible Python Version

**Error:** Package installation fails

**Solution:**

Ensure you're using Python 3.8 or newer:
```bash
python --version
```

If older, upgrade Python or use a virtual environment with a newer version.

---

## Document Ingestion Issues

### No Text Extracted from Document

**Symptoms:**
- `chunks_added: 0` in response
- Warning: "No text extracted from file"

**Causes and Solutions:**

**1. Scanned PDF (Image-based)**
- PDF contains images, not text
- Solution: Use OCR tools (pytesseract, Adobe Acrobat) to convert first

**2. Password-Protected Document**
- Document is encrypted
- Solution: Remove password protection before ingestion

**3. Corrupted File**
- File is damaged or incomplete
- Solution: Re-download or obtain a clean copy

**4. Unsupported Format**
- File extension doesn't match content
- Solution: Verify file format and use correct extension

**5. Empty Document**
- Document actually has no content
- Solution: Check the source document

---

### Slow Document Processing

**Symptoms:**
- Processing takes a very long time
- System becomes unresponsive

**Solutions:**

**1. Reduce Chunk Size**
Edit `server.py`:
```python
CHUNK_SIZE = 2048  # Reduce from 4096
```

**2. Reduce Concurrent Workers**
Edit `server.py`:
```python
MAX_WORKERS = 2  # Reduce from default
```

**3. Process Smaller Batches**
Instead of:
```python
ingest_document("/folder/with/1000/files/")
```

Process in batches:
```python
ingest_document("/folder/batch1/")
ingest_document("/folder/batch2/")
# etc.
```

---

### ChromaDB Errors

**Error:** Database corruption, permission issues

**Solution:**

**1. Clear and Reset Database**
```python
# Using MCP tool
clear_db()
```

Or manually delete:
```bash
# Windows
rmdir /s /q chroma_db

# Mac/Linux
rm -rf chroma_db
```

**2. Check Permissions**
Ensure write permissions for the `chroma_db` directory.

---

## Performance Optimization

### Slow Retrieval

**Solutions:**

1. **Reduce number of results:**
```python
retrieve("query", n=3)  # Instead of n=10
```

2. **Optimize chunk size** - smaller chunks = faster retrieval:
```python
CHUNK_SIZE = 2048
```

3. **Index fewer documents** - only keep relevant documents

---

### High Memory Usage

**Solutions:**

1. **Clear unused collections:**
```python
clear_db()
```

2. **Use smaller embedding model** - consider alternatives to nomic-embed-text

3. **Reduce chunk overlap:**
```python
CHUNK_OVERLAP = 0  # No overlap
```

---

## Network Issues

### Cannot Download from URLs

**Error:** Download failed, connection timeout

**Solutions:**

1. **Check internet connection**
2. **Verify URL is accessible:**
```bash
curl -I https://example.com/document.pdf
```

3. **Download manually** and ingest from local file:
```python
ingest_document("/path/to/downloaded/file.pdf")
```

4. **Check for redirects or authentication requirements**

---

## Getting Help

### Debug Mode

Enable detailed logging by editing `server.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Diagnostic Information

Gather this information when reporting issues:

1. **System Info:**
```bash
python --version
python check_ollama.py
```

2. **Ollama Info:**
```bash
ollama list
curl http://localhost:11434/api/tags
```

3. **Database Info:**
```python
db_info()
```

4. **Error Logs:** Copy complete error messages and stack traces

### Common Log Messages

**INFO: "✓ Ollama connection validated successfully"**
- Good! Everything is working

**ERROR: "Cannot connect to Ollama service"**
- Ollama is not running - start it

**ERROR: "Model 'nomic-embed-text:latest' not found"**
- Model not installed - run `ollama pull nomic-embed-text`

**WARNING: "No text extracted from file.pdf"**
- Check if PDF is scanned/image-based or corrupted

**ERROR: "Connection closed during embedding test"**
- Ollama crashed - check system resources and restart

---

## Prevention Best Practices

1. **Always run check_ollama.py before starting the server**
2. **Monitor system resources** when processing large batches
3. **Start with small test files** before bulk ingestion
4. **Keep Ollama updated** to latest version
5. **Ensure adequate RAM** (8GB+ recommended)
6. **Use SSD storage** for better performance
7. **Close unnecessary applications** when processing large documents

---

## Still Having Issues?

1. Check the [README.md](README.md) for setup instructions
2. Review the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for API changes
3. Run `python check_ollama.py` for detailed diagnostics
4. Check Ollama logs (location varies by OS)
5. Report issues with full diagnostic output
