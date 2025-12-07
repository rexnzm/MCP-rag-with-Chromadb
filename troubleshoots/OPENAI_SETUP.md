# Using OpenAI Embeddings - Setup Guide

This guide shows you how to set up and use OpenAI embeddings instead of Ollama. OpenAI embeddings work reliably in Claude Desktop/MCP context and avoid the connection issues that can occur with Ollama.

## Why OpenAI Embeddings?

**Advantages:**
- Works reliably in Claude Desktop/MCP context (no port conflicts)
- No local installation required
- High-quality embeddings from state-of-the-art models
- Batch processing support (process up to 100 docs at once)
- Faster processing for large document sets
- No system resource requirements (RAM, CPU)

**Considerations:**
- Requires OpenAI API key (costs money, but very affordable)
- Requires internet connection
- Usage-based pricing

## Cost Estimate

OpenAI embedding costs are very reasonable:

**text-embedding-3-small** (recommended):
- $0.02 per 1 million tokens
- ~750 words = 1,000 tokens
- Example: 100 PDF pages (~75,000 words) costs approximately $2.00

**text-embedding-3-large** (higher quality):
- $0.13 per 1 million tokens
- Same as above but ~$13 for 100 PDF pages

**See latest pricing:** https://openai.com/pricing

## Quick Setup (5 minutes)

### 1. Get an OpenAI API Key

1. Go to https://platform.openai.com/signup
2. Create an account (or sign in)
3. Add payment method at https://platform.openai.com/account/billing
4. Go to https://platform.openai.com/api-keys
5. Click "Create new secret key"
6. Name it something like "MCP RAG Server"
7. Copy the key (starts with sk-...)

**Important:** Save this key securely - you cannot view it again!

### 2. Configure Your Server

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit .env and add your API key:
   ```bash
   # On Windows
   notepad .env

   # On Mac/Linux
   nano .env
   ```

3. Update the file:
   ```
   EMBEDDING_PROVIDER=openai
   OPENAI_API_KEY=sk-your-actual-api-key-here
   OPENAI_EMBED_MODEL=text-embedding-3-small
   ```

4. Save and close the file

### 3. Test the Setup

```bash
python server.py
```

You should see:
```
INFO:langchain_vector_db:Using OpenAI embeddings
INFO:langchain_vector_db:Initialized DirectOpenAIEmbeddings with model=text-embedding-3-small, dimensions=1536
```

Your server is now using OpenAI embeddings!

## Model Selection

### text-embedding-3-small (Recommended)

**Best for:** Most use cases

**Specs:**
- Dimensions: 1536
- Cost: $0.02 / 1M tokens
- Speed: Fast
- Quality: Excellent

**Use when:**
- You want good quality at low cost
- Processing large document sets
- Cost is a consideration

### text-embedding-3-large (Premium)

**Best for:** Maximum quality

**Specs:**
- Dimensions: 3072
- Cost: $0.13 / 1M tokens
- Speed: Moderate
- Quality: Best-in-class

**Use when:**
- You need the highest quality retrieval
- Working with technical/specialized content
- Cost is less important than accuracy

**To switch models:**
```
# In .env file
OPENAI_EMBED_MODEL=text-embedding-3-large
```

## Switching Between OpenAI and Ollama

You can switch between providers by changing your .env file:

### Switch to OpenAI:
```
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key...
```

### Switch to Ollama:
```
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

**Important:** After switching providers, you need to **re-ingest your documents** because embeddings from different models are not compatible.

## Usage Examples

### Ingest Documents with OpenAI Embeddings

Everything works the same as before:

```python
# From Claude Desktop or MCP client

# Ingest a single file
await ingest_document("path/to/document.pdf")

# Ingest a directory
await ingest_document("path/to/documents/")

# Ingest from URL
await ingest_document("https://example.com/paper.pdf")

# Search/retrieve
results = await retrieve("your search query", n=5)
```

### Monitor Usage

The server logs embedding operations:

```
INFO:langchain_vector_db:Embedded 100/500 documents
INFO:langchain_vector_db:Successfully embedded 500 documents
```

For batches:
- OpenAI processes up to 100 documents per API call
- Much faster than Ollama (one-at-a-time processing)
- Automatic batching handled by DirectOpenAIEmbeddings

## Troubleshooting

### Error: "OPENAI_API_KEY not set!"

**Solution:**
1. Make sure .env file exists in project root
2. Verify OPENAI_API_KEY is set correctly (no quotes needed)
3. Restart the server after changing .env

### Error: "Invalid OpenAI API key"

**Possible causes:**
- API key is incorrect
- API key was revoked
- Account has billing issues

**Solution:**
1. Verify key at https://platform.openai.com/api-keys
2. Check billing at https://platform.openai.com/account/billing
3. Generate a new key if needed

### Error: "OpenAI API rate limit exceeded"

**Cause:** Sending too many requests too quickly

**Solution:**
1. Wait a minute and try again
2. Process smaller batches of documents
3. Check rate limits at https://platform.openai.com/account/rate-limits
4. Upgrade your account tier if needed

### Error: "Cannot connect to OpenAI API"

**Possible causes:**
- No internet connection
- Firewall blocking requests
- OpenAI service is down

**Solution:**
1. Check internet connection
2. Check OpenAI status: https://status.openai.com
3. Temporarily disable firewall/VPN

## Performance Comparison

| Aspect | OpenAI | Ollama |
|--------|--------|--------|
| **Setup** | 5 minutes | 30+ minutes |
| **MCP Reliability** | Excellent | Can have issues |
| **Speed (100 docs)** | ~10 seconds | ~60 seconds |
| **Batch Processing** | Yes (100/call) | No |
| **Cost** | ~$0.02/1M tokens | Free |
| **Internet Required** | Yes | No |
| **Resource Usage** | None | ~4GB RAM |

## Security Best Practices

### Protect Your API Key

**DO:**
- Store in .env file
- Add .env to .gitignore
- Use separate keys for dev/prod
- Rotate keys periodically
- Set usage limits on OpenAI dashboard

**DO NOT:**
- Commit .env to git
- Share keys in chat/email
- Use same key across projects
- Hard-code keys in source code

### Monitor Usage

1. Go to https://platform.openai.com/usage
2. Set up budget alerts
3. Review usage regularly
4. Set spending limits

## Migration from Ollama

### Option 1: Fresh Start (Recommended)

```bash
# 1. Clear existing data (use the clear_db MCP tool)

# 2. Update .env to use OpenAI
# Edit .env: EMBEDDING_PROVIDER=openai

# 3. Restart server
python server.py

# 4. Re-ingest documents
# Use ingest_document() to add your documents again
```

### Option 2: Side-by-Side

Keep both databases:

```bash
# Rename old database
mv chroma_db chroma_db_ollama

# Configure OpenAI in .env
# Server will create new chroma_db with OpenAI embeddings
```

## FAQ

**Q: Will my existing data work with OpenAI embeddings?**
A: No, you need to re-ingest documents. Different models create incompatible embeddings.

**Q: Can I use both OpenAI and Ollama?**
A: Not simultaneously on the same database. Choose one provider at a time.

**Q: How much will it cost for my use case?**
A: Estimate: (total words / 750) * 0.001 * $0.02 for text-embedding-3-small

**Q: Is my data sent to OpenAI?**
A: Yes, text is sent to OpenAI's API for embedding generation. See: https://openai.com/policies/privacy-policy

**Q: Can I use OpenAI embeddings offline?**
A: No, OpenAI embeddings require internet connection.

**Q: What happens if I run out of API credits?**
A: Embedding operations will fail. Add funds at https://platform.openai.com/account/billing

## Getting Help

If you encounter issues:

1. Check this guide
2. Review server logs for specific errors
3. Verify API key is correct and has credits
4. Check OpenAI status page
5. Review TROUBLESHOOTING.md for general issues

## Next Steps

- Start ingesting documents with OpenAI embeddings
- Monitor your usage on OpenAI dashboard
- Experiment with different models
- Set up usage alerts and spending limits
- Enjoy reliable embedding generation in Claude Desktop!
