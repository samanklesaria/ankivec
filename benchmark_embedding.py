import sqlite3
import time
import ollama
from chromadb.utils import embedding_functions
import onnxruntime as ort

def benchmark_ollama(notes):
    note_ids, card_text = zip(*notes)
    joined_text = ["search_document: " + " ".join(c.split(chr(0x1f))) for c in card_text]

    start = time.time()
    embeddings = ollama.embed(model='chroma/all-minilm-l6-v2-f32', input=joined_text)["embeddings"]
    end = time.time()

    return end - start, len(embeddings)

def benchmark_chromadb(notes, providers=None):
    _, card_text = zip(*notes)
    joined_text = ["search_document: " + " ".join(c.split(chr(0x1f))) for c in card_text]

    if providers:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        ef = ONNXMiniLM_L6_V2(preferred_providers=providers)
    else:
        ef = embedding_functions.DefaultEmbeddingFunction()

    start = time.time()
    embeddings = ef(joined_text)
    end = time.time()

    return end - start, len(embeddings)

if __name__ == "__main__":
    db_path = "/Users/sam/Library/Application Support/Anki2/User 1/collection.anki2"
    conn = sqlite3.connect(db_path)

    notes = list(conn.cursor().execute("SELECT id, flds FROM notes ORDER BY mod DESC LIMIT 256"))

    if len(notes) == 0:
        print("No notes found in database")
        conn.close()
        exit(1)

    print(f"Benchmarking with {len(notes)} notes")
    print("=" * 60 + "\n")

    print("ONNX Runtime Info:")
    available_providers = ort.get_available_providers()
    print(f"  Available providers: {', '.join(available_providers)}")
    has_coreml = 'CoreMLExecutionProvider' in available_providers
    print(f"  Metal/CoreML: {'✓ available' if has_coreml else '✗ not available'}")
    print()

    print("Running Ollama benchmark...")
    ollama_time, ollama_count = benchmark_ollama(notes)
    print(f"  Time: {ollama_time:.3f}s")
    print(f"  Embeddings: {ollama_count}")
    print(f"  Rate: {ollama_count/ollama_time:.1f} embeddings/sec\n")

    print("Running ChromaDB DefaultEmbeddingFunction benchmark (CPU)...")
    chroma_time, chroma_count = benchmark_chromadb(notes)
    print(f"  Time: {chroma_time:.3f}s")
    print(f"  Embeddings: {chroma_count}")
    print(f"  Rate: {chroma_count/chroma_time:.1f} embeddings/sec\n")

    if has_coreml:
        print("Running ChromaDB with CoreML/Metal benchmark...")
        chroma_metal_time, chroma_metal_count = benchmark_chromadb(notes, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        print(f"  Time: {chroma_metal_time:.3f}s")
        print(f"  Embeddings: {chroma_metal_count}")
        print(f"  Rate: {chroma_metal_count/chroma_metal_time:.1f} embeddings/sec\n")

    print("=" * 60)
    print("Comparison (vs Ollama):")
    speedup = chroma_time / ollama_time
    if speedup > 1:
        print(f"  ChromaDB (CPU): Ollama is {speedup:.2f}x faster")
    else:
        print(f"  ChromaDB (CPU): {1/speedup:.2f}x faster than Ollama")

    if has_coreml:
        speedup_metal = chroma_metal_time / ollama_time
        if speedup_metal > 1:
            print(f"  ChromaDB (Metal): Ollama is {speedup_metal:.2f}x faster")
        else:
            print(f"  ChromaDB (Metal): {1/speedup_metal:.2f}x faster than Ollama")

    conn.close()
