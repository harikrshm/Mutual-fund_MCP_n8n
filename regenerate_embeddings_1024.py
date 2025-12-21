"""
Regenerate embeddings with a model that produces 1024 dimensions
"""
import json
from sentence_transformers import SentenceTransformer
import time

def regenerate_embeddings(input_file: str, output_file: str, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Regenerate embeddings with a different model
    
    Note: all-mpnet-base-v2 produces 768 dimensions
    For 1024 dimensions, we might need to use OpenAI or another service
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Load existing JSON
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vectors = data.get('vectors', [])
    print(f"Found {len(vectors)} vectors to regenerate")
    
    # Check current dimension
    if vectors:
        current_dim = len(vectors[0]['values'])
        print(f"Current embedding dimension: {current_dim}")
    
    # Regenerate embeddings
    print(f"\nRegenerating embeddings with {model_name}...")
    print("(This model produces 768 dimensions, not 1024)")
    print("For 1024 dimensions, you may need to use OpenAI's text-embedding-3-small model")
    
    new_vectors = []
    for i, vec in enumerate(vectors, 1):
        text = vec['metadata']['text']
        print(f"Processing vector {i}/{len(vectors)}...")
        
        # Generate new embedding
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        
        # Update vector
        new_vec = {
            'id': vec['id'],
            'values': embedding,
            'metadata': vec['metadata']
        }
        new_vectors.append(new_vec)
        
        time.sleep(0.1)  # Rate limiting
    
    # Update data
    data['vectors'] = new_vectors
    new_dim = len(new_vectors[0]['values']) if new_vectors else 0
    data['metadata']['embedding_dimension'] = new_dim
    data['metadata']['embedding_model'] = model_name
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! New embedding dimension: {new_dim}")
    print(f"Note: Your index expects 1024 dimensions.")
    print(f"Options:")
    print(f"  1. Use OpenAI's text-embedding-3-small (1536 dims) or text-embedding-3-large (3072 dims)")
    print(f"  2. Recreate your Pinecone index with dimension {new_dim}")

if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Regeneration Script")
    print("=" * 60)
    print("\nYour Pinecone index expects 1024 dimensions,")
    print("but current embeddings are 384 dimensions.")
    print("\nOptions:")
    print("1. Regenerate with all-mpnet-base-v2 (768 dims) - closer to 1024")
    print("2. Use OpenAI API for 1024+ dimensions")
    print("3. Recreate Pinecone index with 384 dimensions")
    print("\n" + "=" * 60)
    
    # For now, let's use a workaround - we'll need to either:
    # 1. Use OpenAI for 1024 dims
    # 2. Or recreate index with 384 dims
    
    print("\nSince sentence-transformers don't have a 1024-dim model,")
    print("you have two options:")
    print("\nOption A: Recreate Pinecone index with 384 dimensions")
    print("Option B: Use OpenAI API to generate 1024-dim embeddings")
    
    choice = input("\nChoose option (A/B): ").strip().upper()
    
    if choice == 'A':
        print("\nYou'll need to recreate your Pinecone index with 384 dimensions.")
        print("The current embeddings (384 dims) will work with a 384-dim index.")
    elif choice == 'B':
        print("\nTo use OpenAI, you'll need to modify the scraper to use OpenAI API.")
        print("Would you like me to create a script that uses OpenAI for 1024-dim embeddings?")
    else:
        print("Invalid choice. Exiting.")

