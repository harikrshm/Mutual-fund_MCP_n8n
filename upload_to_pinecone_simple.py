"""
Script to upload Pinecone-compatible JSON file to Pinecone vector database
Usage: python upload_to_pinecone_simple.py <api_key> <index_name> [json_file] [namespace]
"""
import json
import sys
import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import time
import io
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load the Pinecone JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def upload_to_pinecone(
    api_key: str,
    index_name: str,
    vectors: List[Dict[str, Any]],
    namespace: str = None,
    batch_size: int = 100
):
    """Upload vectors to Pinecone"""
    # Initialize Pinecone client
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    print(f"Checking index: {index_name}")
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"\nIndex '{index_name}' does not exist.")
        print(f"Available indexes: {existing_indexes}")
        print(f"\nCreating index '{index_name}'...")
        
        # Get dimension from first vector
        dimension = len(vectors[0]['values']) if vectors else 384
        print(f"Creating index with dimension {dimension}...")
        
        # Create index (using ServerlessSpec)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change if needed
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while index_name not in [index.name for index in pc.list_indexes()]:
            time.sleep(2)
        time.sleep(5)  # Additional wait
        print("Index created successfully!")
    
    # Connect to the index
    print(f"\nConnecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Current index stats: {stats}")
    
    # Upload vectors in batches
    total_vectors = len(vectors)
    print(f"\nUploading {total_vectors} vectors to Pinecone...")
    print(f"Using namespace: {namespace if namespace else 'default'}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    
    uploaded = 0
    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_vectors + batch_size - 1) // batch_size
        
        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} vectors)...")
        
        try:
            # Format vectors for Pinecone v8 API
            formatted_batch = []
            for vec in batch:
                formatted_vec = {
                    "id": vec["id"],
                    "values": vec["values"],
                    "metadata": vec.get("metadata", {})
                }
                formatted_batch.append(formatted_vec)
            
            # Upload batch
            if namespace:
                index.upsert(vectors=formatted_batch, namespace=namespace)
            else:
                index.upsert(vectors=formatted_batch)
            
            uploaded += len(batch)
            print(f"  [OK] Uploaded {uploaded}/{total_vectors} vectors")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  X Error uploading batch: {str(e)}")
            raise
    
    print("-" * 60)
    print(f"\n[SUCCESS] Successfully uploaded {uploaded} vectors to Pinecone!")
    
    # Get updated stats
    stats = index.describe_index_stats()
    print(f"\nUpdated index stats:")
    if namespace:
        ns_stats = stats.namespaces.get(namespace, {})
        print(f"  Namespace '{namespace}': {ns_stats.get('vector_count', 0)} vectors")
    else:
        print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Index dimension: {stats.dimension}")

def main():
    """Main function"""
    print("=" * 60)
    print("Pinecone Vector Database Uploader")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("\nUsage: python upload_to_pinecone_simple.py <api_key> <index_name> [json_file] [namespace]")
        print("\nExample:")
        print("  python upload_to_pinecone_simple.py your-api-key your-index-name")
        print("  python upload_to_pinecone_simple.py your-api-key your-index-name pinecone_vectors_groww.json")
        print("  python upload_to_pinecone_simple.py your-api-key your-index-name pinecone_vectors_groww.json mutual-fund-docs")
        sys.exit(1)
    
    api_key = sys.argv[1]
    index_name = sys.argv[2]
    json_file = sys.argv[3] if len(sys.argv) > 3 else "pinecone_vectors_groww.json"
    namespace = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        sys.exit(1)
    
    # Load JSON file
    print(f"\nLoading JSON file: {json_file}")
    try:
        data = load_json_file(json_file)
        vectors = data.get('vectors', [])
        file_namespace = data.get('namespace', None)
        
        if not vectors:
            print("Error: No vectors found in JSON file!")
            sys.exit(1)
        
        print(f"Loaded {len(vectors)} vectors from JSON file")
        
        # Use namespace from file if not provided
        if not namespace and file_namespace:
            namespace = file_namespace
            print(f"Using namespace from file: {namespace}")
        
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        sys.exit(1)
    
    # Upload to Pinecone
    print("\n" + "=" * 60)
    try:
        upload_to_pinecone(
            api_key=api_key,
            index_name=index_name,
            vectors=vectors,
            namespace=namespace,
            batch_size=100
        )
        print("\n" + "=" * 60)
        print("Upload completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Error during upload: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

