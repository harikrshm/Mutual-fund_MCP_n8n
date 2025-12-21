"""
Script to upload Pinecone-compatible JSON file to Pinecone vector database
"""
import json
import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import time

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load the Pinecone JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with vectors and metadata
    """
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
    """
    Upload vectors to Pinecone
    
    Args:
        api_key: Pinecone API key
        index_name: Name of the Pinecone index
        vectors: List of vectors to upload
        namespace: Namespace in Pinecone (optional)
        batch_size: Number of vectors to upload per batch
    """
    # Initialize Pinecone client
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    print(f"Checking index: {index_name}")
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"\nIndex '{index_name}' does not exist.")
        print(f"Available indexes: {existing_indexes}")
        create = input(f"\nWould you like to create index '{index_name}'? [y/N]: ").strip().lower()
        
        if create == 'y':
            # Get dimension from first vector
            dimension = len(vectors[0]['values']) if vectors else 384
            print(f"Creating index '{index_name}' with dimension {dimension}...")
            
            # Create index (using ServerlessSpec - adjust if using different spec)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Change to your preferred region
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while index_name not in [index.name for index in pc.list_indexes()]:
                time.sleep(1)
            time.sleep(5)  # Additional wait for index to be fully ready
            print("Index created successfully!")
        else:
            print("Exiting. Please create the index manually or provide an existing index name.")
            return
    
    # Connect to the index
    print(f"\nConnecting to index: {index_name}")
    index = pc.Index(index_name)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
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
            # Pinecone expects: [{"id": "...", "values": [...], "metadata": {...}}, ...]
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
            print(f"  ✓ Uploaded {uploaded}/{total_vectors} vectors")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error uploading batch: {str(e)}")
            raise
    
    print("-" * 60)
    print(f"\n✓ Successfully uploaded {uploaded} vectors to Pinecone!")
    
    # Get updated stats
    stats = index.describe_index_stats()
    print(f"\nUpdated index stats:")
    if namespace:
        print(f"  Namespace '{namespace}': {stats.namespaces.get(namespace, {}).get('vector_count', 0)} vectors")
    else:
        print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Index dimension: {stats.dimension}")
    print(f"  Index fullness: {stats.index_fullness}")

def main():
    """
    Main function to upload JSON file to Pinecone
    """
    print("=" * 60)
    print("Pinecone Vector Database Uploader")
    print("=" * 60)
    
    # Get JSON file path
    json_file = input("\nEnter JSON file path (default: pinecone_vectors_groww.json): ").strip()
    if not json_file:
        json_file = "pinecone_vectors_groww.json"
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        return
    
    # Load JSON file
    print(f"\nLoading JSON file: {json_file}")
    try:
        data = load_json_file(json_file)
        vectors = data.get('vectors', [])
        namespace = data.get('namespace', None)
        
        if not vectors:
            print("Error: No vectors found in JSON file!")
            return
        
        print(f"Loaded {len(vectors)} vectors from JSON file")
        if namespace:
            print(f"Namespace from file: {namespace}")
        
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return
    
    # Get Pinecone credentials
    print("\n" + "-" * 60)
    print("Pinecone Configuration")
    print("-" * 60)
    
    api_key = input("Enter your Pinecone API key: ").strip()
    if not api_key:
        print("Error: API key is required!")
        return
    
    index_name = input("Enter Pinecone index name: ").strip()
    if not index_name:
        print("Error: Index name is required!")
        return
    
    # Ask about namespace
    use_namespace = input(f"\nUse namespace '{namespace}' from file? [Y/n]: ").strip().lower()
    if use_namespace == 'n':
        custom_namespace = input("Enter custom namespace (or press Enter for default): ").strip()
        namespace = custom_namespace if custom_namespace else None
    elif not namespace:
        namespace = None
    
    # Ask about batch size
    batch_size_input = input("\nEnter batch size (default: 100): ").strip()
    batch_size = int(batch_size_input) if batch_size_input.isdigit() else 100
    
    # Upload to Pinecone
    print("\n" + "=" * 60)
    try:
        upload_to_pinecone(
            api_key=api_key,
            index_name=index_name,
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size
        )
        print("\n" + "=" * 60)
        print("Upload completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error during upload: {str(e)}")
        print("Please check your API key, index name, and network connection.")

if __name__ == "__main__":
    main()

