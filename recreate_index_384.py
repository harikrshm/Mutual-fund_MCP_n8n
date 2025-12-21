"""
Recreate Pinecone index with 384 dimensions and upload vectors
"""
import json
import sys
import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import time

def recreate_index_and_upload(
    api_key: str,
    index_name: str,
    json_file: str,
    namespace: str = None
):
    """Recreate index with 384 dimensions and upload vectors"""
    
    # Initialize Pinecone client
    print("=" * 60)
    print("Recreating Pinecone Index with 384 Dimensions")
    print("=" * 60)
    print(f"\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Load JSON file
    print(f"Loading JSON file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vectors = data.get('vectors', [])
    if not vectors:
        print("Error: No vectors found in JSON file!")
        return
    
    file_namespace = data.get('namespace', None)
    if not namespace and file_namespace:
        namespace = file_namespace
    
    print(f"Loaded {len(vectors)} vectors")
    print(f"Vector dimension: {len(vectors[0]['values'])}")
    print(f"Namespace: {namespace if namespace else 'default'}")
    
    # Check if index exists
    print(f"\nChecking index: {index_name}")
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' exists. Deleting it...")
        pc.delete_index(index_name)
        
        # Wait for deletion to complete
        print("Waiting for index deletion...")
        while index_name in [index.name for index in pc.list_indexes()]:
            time.sleep(2)
        print("Index deleted successfully!")
    
    # Create new index with 384 dimensions
    print(f"\nCreating new index '{index_name}' with 384 dimensions...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
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
    print(f"Index stats: {stats}")
    
    # Upload vectors in batches
    total_vectors = len(vectors)
    batch_size = 100
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
            print(f"  [ERROR] Error uploading batch: {str(e)}")
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
    print(f"  Index metric: {stats.metric}")

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python recreate_index_384.py <api_key> <index_name> [json_file] [namespace]")
        print("\nExample:")
        print("  python recreate_index_384.py your-api-key mutualfundfees")
        print("  python recreate_index_384.py your-api-key mutualfundfees pinecone_vectors_groww.json")
        sys.exit(1)
    
    api_key = sys.argv[1]
    index_name = sys.argv[2]
    json_file = sys.argv[3] if len(sys.argv) > 3 else "pinecone_vectors_groww.json"
    namespace = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        sys.exit(1)
    
    try:
        recreate_index_and_upload(api_key, index_name, json_file, namespace)
        print("\n" + "=" * 60)
        print("Index recreated and upload completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

