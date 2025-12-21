"""
Regenerate embeddings using OpenAI API to match 1024 dimensions required by Pinecone index
"""
import json
import os
from openai import OpenAI
import time

def regenerate_embeddings_openai(input_file: str, output_file: str, api_key: str):
    """
    Regenerate embeddings using OpenAI's text-embedding-3-small model
    Configured to produce 1024 dimensions to match Pinecone index
    """
    client = OpenAI(api_key=api_key)
    
    # Load existing JSON
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vectors = data.get('vectors', [])
    print(f"Found {len(vectors)} vectors to regenerate")
    
    if not vectors:
        print("No vectors found!")
        return
    
    # Check current dimension
    current_dim = len(vectors[0]['values'])
    print(f"Current embedding dimension: {current_dim}")
    print(f"Target dimension: 1024")
    
    # Regenerate embeddings
    print(f"\nRegenerating embeddings with OpenAI text-embedding-3-small (1024 dims)...")
    print("-" * 60)
    
    new_vectors = []
    for i, vec in enumerate(vectors, 1):
        text = vec['metadata']['text']
        print(f"Processing vector {i}/{len(vectors)}...", end='\r')
        
        try:
            # Generate embedding with 1024 dimensions
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1024  # Specify 1024 dimensions
            )
            
            embedding = response.data[0].embedding
            
            # Update vector
            new_vec = {
                'id': vec['id'],
                'values': embedding,
                'metadata': vec['metadata']
            }
            new_vectors.append(new_vec)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"\nError processing vector {i}: {str(e)}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"Processed {len(new_vectors)} vectors")
    
    # Update data
    data['vectors'] = new_vectors
    new_dim = len(new_vectors[0]['values']) if new_vectors else 0
    data['metadata']['embedding_dimension'] = new_dim
    data['metadata']['embedding_model'] = 'text-embedding-3-small'
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! New embedding dimension: {new_dim}")
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Regenerate Embeddings for 1024-Dimension Pinecone Index")
    print("=" * 60)
    
    input_file = "pinecone_vectors_groww.json"
    output_file = "pinecone_vectors_groww_1024.json"
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if not api_key:
            print("Error: API key is required!")
            exit(1)
    
    regenerate_embeddings_openai(input_file, output_file, api_key)
    print("\n" + "=" * 60)
    print("You can now upload the new file to Pinecone:")
    print(f"python upload_to_pinecone_simple.py <api_key> mutualfundfees {output_file}")
    print("=" * 60)

