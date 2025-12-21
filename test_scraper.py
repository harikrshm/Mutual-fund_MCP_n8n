"""
Quick test script for the scraper with AMFI PDF URLs
"""
from scraper import WebScraperToPinecone

def test_amfi_urls():
    """Test the scraper with AMFI PDF URLs"""
    
    # Groww HTML URLs to test
    test_urls = [
        'https://groww.in/blog/mutual-fund-fees-and-charges?utm_source=chatgpt.com',
        'https://groww.in/p/exit-load-in-mutual-funds?utm_source=chatgpt.com'
    ]
    
    print("=" * 60)
    print("Testing Web Scraper with Groww HTML URLs")
    print("Using Sentence Transformers for Embeddings (No API Key Required!)")
    print("=" * 60)
    print(f"\nTesting {len(test_urls)} URLs:")
    for i, url in enumerate(test_urls, 1):
        print(f"  {i}. {url}")
    
    # Default to generating embeddings (no API key needed)
    use_embeddings = True
    
    print("\nUsing embedding model: all-MiniLM-L6-v2")
    print("(First run will download the model, subsequent runs will be faster)")
    
    # Initialize scraper
    scraper = WebScraperToPinecone(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=2000,
        chunk_overlap=200
    )
    
    # Scrape URLs
    print(f"\nStarting to scrape {len(test_urls)} URLs...")
    print("-" * 60)
    scraper.scrape_urls(test_urls, generate_embeddings=use_embeddings)
    
    # Save to JSON file
    output_file = 'pinecone_vectors_groww.json'
    namespace = 'mutual-fund-docs'
    
    print("\n" + "-" * 60)
    scraper.save_to_pinecone_json(output_file, namespace)
    
    print(f"\n{'=' * 60}")
    print(f"Done! Check {output_file} for the Pinecone-compatible JSON file.")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    test_amfi_urls()

