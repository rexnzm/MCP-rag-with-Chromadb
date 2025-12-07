"""
Test script to verify the custom DirectOllamaEmbeddings implementation.
Run this before starting the MCP server to ensure embeddings work correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the custom embeddings class
sys.path.insert(0, str(Path(__file__).parent))
from server import DirectOllamaEmbeddings, EMBED_MODEL, OLLAMA_BASE_URL

def test_embeddings():
    """Test the DirectOllamaEmbeddings implementation."""
    print("=" * 60)
    print("Testing DirectOllamaEmbeddings")
    print("=" * 60)
    print(f"Model: {EMBED_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print()

    try:
        # Initialize embeddings
        print("1. Initializing embeddings...")
        embeddings = DirectOllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        print("   ✓ Embeddings initialized")
        print()

        # Test single query embedding
        print("2. Testing single query embedding...")
        test_query = "What is the capital of France?"
        query_embedding = embeddings.embed_query(test_query)
        print(f"   Query: '{test_query}'")
        print(f"   ✓ Embedding generated: {len(query_embedding)} dimensions")
        print(f"   Sample values: {query_embedding[:5]}")
        print()

        # Test document embeddings (batch)
        print("3. Testing batch document embeddings...")
        test_docs = [
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris.",
            "France is a country in Europe."
        ]
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"   Documents: {len(test_docs)}")
        print(f"   ✓ Generated {len(doc_embeddings)} embeddings")
        for i, emb in enumerate(doc_embeddings):
            print(f"     Doc {i+1}: {len(emb)} dimensions")
        print()

        # Test callable interface (ChromaDB compatibility)
        print("4. Testing callable interface (ChromaDB)...")
        callable_embedding = embeddings("Test text for callable interface")
        print(f"   ✓ Callable works: {len(callable_embedding)} dimensions")
        print()

        # Verify all embeddings have same dimension
        print("5. Verifying embedding dimensions...")
        all_dims = [len(query_embedding)] + [len(e) for e in doc_embeddings] + [len(callable_embedding)]
        if len(set(all_dims)) == 1:
            print(f"   ✓ All embeddings have consistent dimension: {all_dims[0]}")
        else:
            print(f"   ✗ Inconsistent dimensions: {all_dims}")
            return False
        print()

        # Success!
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("The DirectOllamaEmbeddings implementation is working correctly!")
        print("You can now start the MCP server with confidence.")
        print()
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Please ensure:")
        print("1. Ollama is running: http://localhost:11434")
        print("2. Model is installed: ollama pull nomic-embed-text")
        print("3. Run: python check_ollama.py for detailed diagnostics")
        print()
        return False


if __name__ == "__main__":
    success = test_embeddings()
    sys.exit(0 if success else 1)
