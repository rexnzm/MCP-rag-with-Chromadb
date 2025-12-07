# Migration Guide: Multi-Format Document Support

This guide explains the changes made to add multi-format document support to the MCP RAG ChromaDB server.

## What's New

The server has been refactored from PDF-only support to comprehensive multi-format document ingestion, including:
- PDF, DOCX, PPTX, XLSX, ODT, HTML, TXT, MD, CSV, JSON, XML, and more
- Automatic format detection based on file extensions
- Graceful handling of missing optional dependencies
- Improved error handling and logging

## Breaking Changes

**None!** All changes are backwards compatible. The legacy `ingest_pdf` tool still works and redirects to the new `ingest_document` tool.

## API Changes

### New MCP Tool: `ingest_document`

This is the new primary tool for document ingestion. It replaces `ingest_pdf` but maintains the same API:

```python
# Old way (still works)
await ingest_pdf("document.pdf")

# New way (recommended)
await ingest_document("document.pdf")
await ingest_document("document.docx")
await ingest_document("document.xlsx")
await ingest_document("/path/to/documents/")  # All formats
```

### Updated Functions

| Old Function | New Function | Status |
|-------------|--------------|--------|
| `extract_text_from_pdf()` | `extract_text()` | Both available |
| `pdf_to_chunks()` | `document_to_chunks()` | Both available |
| `download_pdf()` | `download_file()` | Both available |
| `ingest_pdf` (tool) | `ingest_document` (tool) | Both available |

## New Features

### 1. Multi-Format Extraction

The server now includes dedicated extractors for each format:

```python
extract_text_from_pdf(path)      # PDF files
extract_text_from_docx(path)     # Word documents
extract_text_from_pptx(path)     # PowerPoint presentations
extract_text_from_xlsx(path)     # Excel spreadsheets
extract_text_from_odt(path)      # OpenDocument Text
extract_text_from_html(path)     # HTML files
extract_text_from_markdown(path) # Markdown files
extract_text_from_csv(path)      # CSV files
extract_text_from_json(path)     # JSON files
extract_text_from_xml(path)      # XML files
extract_text_from_txt(path)      # Plain text (fallback)
```

### 2. Unified Extraction Interface

The new `extract_text()` function automatically routes to the appropriate extractor:

```python
text = extract_text("document.pdf")    # Uses PDF extractor
text = extract_text("document.docx")   # Uses DOCX extractor
text = extract_text("unknown.xyz")     # Falls back to TXT extractor
```

### 3. Format-Agnostic Processing

The new `document_to_chunks()` function works with any supported format:

```python
# Old way
chunks, ids = pdf_to_chunks("document.pdf")

# New way (works with any format)
chunks, ids = document_to_chunks("document.pdf")
chunks, ids = document_to_chunks("document.docx")
chunks, ids = document_to_chunks("document.xlsx")
```

### 4. Enhanced Metadata

Chunks now include file type information:

```python
{
    "source": "/path/to/document.docx",
    "filename": "document.docx",
    "file_type": ".docx",  # NEW
    "chunk_index": 0,
    "total_chunks": 10
}
```

### 5. Dependency Detection

The server detects missing optional dependencies and logs helpful messages:

```
INFO: PDF: ✓
INFO: DOCX: ✗ (install python-docx)
INFO: HTML: ✓
INFO: PPTX: ✗ (install python-pptx)
```

## Installation Updates

### New Requirements

Install all optional dependencies for full format support:

```bash
pip install -r requirements.txt
```

Or install selectively based on your needs:

```bash
# PDF only
pip install PyPDF2

# Microsoft Office formats
pip install python-docx python-pptx openpyxl

# Web formats
pip install beautifulsoup4 lxml

# OpenDocument formats
pip install odfpy

# Markdown enhancement
pip install markdown
```

## Code Examples

### Before (PDF Only)

```python
# Only worked with PDFs
await ingest_pdf("https://example.com/document.pdf")
await ingest_pdf("/path/to/pdfs/")
```

### After (All Formats)

```python
# Works with any supported format
await ingest_document("https://example.com/document.pdf")
await ingest_document("https://example.com/report.docx")
await ingest_document("https://example.com/presentation.pptx")
await ingest_document("/path/to/documents/")  # Mixed formats

# Legacy tool still works
await ingest_pdf("https://example.com/document.pdf")
```

### Batch Processing Multiple Formats

```python
# Directory with mixed document types
await ingest_document("/path/to/mixed-documents/")
# Processes: .pdf, .docx, .xlsx, .pptx, .html, .txt, .md, etc.
```

## Performance Considerations

### Concurrent Processing

The server processes multiple documents concurrently using ThreadPoolExecutor:

```python
MAX_WORKERS = min(8, (os.cpu_count() or 1) * 2)
```

For large batches, processing time scales with the number of CPU cores.

### Format-Specific Performance

| Format | Speed | Notes |
|--------|-------|-------|
| TXT, MD, CSV, JSON, XML | Fast | Simple parsing |
| PDF | Medium | Depends on PDF complexity |
| DOCX, PPTX, ODT | Medium | Structured formats |
| HTML | Medium | Depends on page complexity |
| XLSX | Medium-Slow | Row iteration |

## Error Handling

### Missing Dependencies

If a required library is missing, the server logs a warning and skips that file:

```
ERROR: python-docx not installed. Cannot process DOCX files.
```

### Unsupported Formats

Unknown file types fall back to text extraction:

```
WARNING: Unsupported file type: .xyz, attempting TXT extraction
```

### Extraction Failures

Individual file failures don't crash the entire batch:

```
WARNING: No text extracted from document.pdf
INFO: Prepared 150 chunks from document.docx
```

## Configuration

### Supported Extensions

You can modify the supported extensions in `server.py`:

```python
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.txt', '.html', '.htm',
    '.pptx', '.ppt', '.xlsx', '.xls', '.odt', '.md',
    '.rtf', '.csv', '.json', '.xml'
}
```

### Chunk Settings

Adjust chunking behavior for different document types:

```python
CHUNK_SIZE = 4096           # Characters per chunk
CHUNK_OVERLAP = 409         # Overlap between chunks
```

## Testing Your Migration

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the server:**
   ```bash
   python server.py
   ```

3. **Verify format support in logs:**
   Look for the "Supported Formats" section in startup logs

4. **Test document ingestion:**
   ```python
   # Test with different formats
   await ingest_document("test.pdf")
   await ingest_document("test.docx")
   await ingest_document("test.xlsx")
   ```

5. **Verify retrieval:**
   ```python
   results = await retrieve("your query")
   # Check that results include metadata with file_type
   ```

## Rollback Plan

If you need to rollback:

1. The original `ingest_pdf`, `pdf_to_chunks`, and `download_pdf` functions still exist
2. All legacy code paths remain functional
3. No database schema changes - existing data remains compatible

## Support

For issues or questions:
1. Check the error logs for missing dependencies
2. Verify file formats are in `SUPPORTED_EXTENSIONS`
3. Ensure Ollama is running and accessible
4. Review the README.md for troubleshooting tips

## Next Steps

After migration, consider:
1. Testing with your most common document formats
2. Adjusting `CHUNK_SIZE` for optimal retrieval
3. Adding custom extractors for specialized formats
4. Monitoring performance with large document sets
