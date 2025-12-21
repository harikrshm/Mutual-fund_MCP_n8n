import requests
from bs4 import BeautifulSoup
import pdfplumber
import json
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import hashlib
import time
from sentence_transformers import SentenceTransformer
import os

class WebScraperToPinecone:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize the scraper
        
        Args:
            embedding_model: Sentence transformer model name (default: "all-MiniLM-L6-v2")
                           Options: "all-MiniLM-L6-v2" (fast, 384 dims), 
                                   "all-mpnet-base-v2" (slower, better quality, 768 dims)
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Embedding model loaded successfully!")
        self.vectors = []
        
    def fetch_url(self, url: str) -> Dict[str, Any]:
        """
        Fetch content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with content type and data
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if it's a PDF
            if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                return {
                    'type': 'pdf',
                    'content': response.content,
                    'url': url
                }
            # Check if it's HTML
            elif 'text/html' in content_type or url.lower().endswith(('.html', '.htm')):
                return {
                    'type': 'html',
                    'content': response.text,
                    'url': url
                }
            else:
                # Try to parse as text
                return {
                    'type': 'text',
                    'content': response.text,
                    'url': url
                }
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract text from HTML content
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content
        
        Args:
            pdf_content: PDF content as bytes
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            # Use BytesIO to handle PDF content
            from io import BytesIO
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                print(f"  PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"  Extracted text from page {i}: {len(page_text)} characters")
                    else:
                        print(f"  Warning: No text found on page {i} (might be image-based/scanned PDF)")
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            print(f"  PDF might be encrypted, corrupted, or image-based. Trying alternative method...")
            # Try alternative: PyPDF2 as fallback
            try:
                import PyPDF2
                from io import BytesIO
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                print(f"  Successfully extracted text using PyPDF2 fallback")
            except Exception as e2:
                print(f"  PyPDF2 also failed: {str(e2)}")
                print(f"  Note: This PDF may require OCR (Optical Character Recognition) for image-based PDFs")
        
        return text.strip()
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vectorization
        
        Args:
            text: Text to chunk
            source: Source URL
            
        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_id = self.generate_chunk_id(source, chunk_index)
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'source': source,
                    'start_char': start_char,
                    'end_char': start_char + len(current_chunk),
                    'chunk_id': chunk_id
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-int(self.chunk_overlap / 10):]  # Approximate word overlap
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                start_char = chunks[-1]['end_char'] - len(' '.join(overlap_words))
                chunk_index += 1
            else:
                current_chunk += sentence + ' '
        
        # Add remaining chunk
        if current_chunk.strip():
            chunk_id = self.generate_chunk_id(source, chunk_index)
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'source': source,
                'start_char': start_char,
                'end_char': start_char + len(current_chunk),
                'chunk_id': chunk_id
            })
        
        return chunks
    
    def generate_chunk_id(self, source: str, chunk_index: int) -> str:
        """
        Generate a unique chunk ID
        
        Args:
            source: Source URL
            chunk_index: Index of the chunk
            
        Returns:
            Unique chunk ID
        """
        # Create a hash from the source URL
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_chunk_{chunk_index}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Sentence Transformers
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Generate embedding using sentence transformers
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []
    
    def process_url(self, url: str, generate_embeddings: bool = True) -> List[Dict[str, Any]]:
        """
        Process a single URL and extract text
        
        Args:
            url: URL to process
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            List of vector dictionaries
        """
        print(f"Processing: {url}")
        
        # Fetch URL
        fetched_data = self.fetch_url(url)
        if not fetched_data:
            return []
        
        # Extract text based on content type
        if fetched_data['type'] == 'pdf':
            text = self.extract_text_from_pdf(fetched_data['content'])
        elif fetched_data['type'] == 'html':
            text = self.extract_text_from_html(fetched_data['content'])
        else:
            text = fetched_data['content']
        
        if not text:
            print(f"No text extracted from {url}")
            return []
        
        print(f"Extracted {len(text)} characters from {url}")
        
        # Chunk the text
        chunks = self.chunk_text(text, url)
        print(f"Created {len(chunks)} chunks from {url}")
        
        # Create vectors
        vectors = []
        for chunk in chunks:
            vector_data = {
                'id': chunk['chunk_id'],
                'metadata': {
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'url': url,
                    'chunk_index': chunk['chunk_index'],
                    'start_char': chunk['start_char'],
                    'end_char': chunk['end_char'],
                    'content_type': fetched_data['type'],
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                }
            }
            
            # Generate embedding if requested
            if generate_embeddings:
                embedding = self.generate_embedding(chunk['text'])
                if embedding:
                    vector_data['values'] = embedding
                    vectors.append(vector_data)
            else:
                # Store without embedding (you can generate embeddings later)
                vectors.append(vector_data)
        
        return vectors
    
    def scrape_urls(self, urls: List[str], generate_embeddings: bool = True) -> None:
        """
        Scrape multiple URLs and collect vectors
        
        Args:
            urls: List of URLs to scrape
            generate_embeddings: Whether to generate embeddings
        """
        for url in urls:
            vectors = self.process_url(url, generate_embeddings)
            self.vectors.extend(vectors)
            time.sleep(1)  # Rate limiting between URLs
    
    def save_to_pinecone_json(self, output_file: str, namespace: str = "mutual-fund-docs") -> None:
        """
        Save vectors to Pinecone-compatible JSON file
        
        Args:
            output_file: Output file path
            namespace: Pinecone namespace
        """
        # Filter vectors that have embeddings
        vectors_with_embeddings = [v for v in self.vectors if 'values' in v]
        
        pinecone_data = {
            'vectors': vectors_with_embeddings,
            'namespace': namespace,
            'metadata': {
                'total_vectors': len(vectors_with_embeddings),
                'total_chunks': len(self.vectors),
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'sources': list(set([v['metadata']['source'] for v in vectors_with_embeddings]))
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pinecone_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(vectors_with_embeddings)} vectors to {output_file}")
        print(f"Total chunks processed: {len(self.vectors)}")
        print(f"Sources: {len(pinecone_data['metadata']['sources'])} unique URLs")


def main():
    """
    Main function to run the scraper
    """
    # Configuration
    OUTPUT_FILE = 'pinecone_vectors.json'
    NAMESPACE = 'mutual-fund-docs'
    
    # Embedding model options
    # "all-MiniLM-L6-v2" - Fast, 384 dimensions, good for most use cases
    # "all-mpnet-base-v2" - Slower, 768 dimensions, better quality
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # List of URLs to scrape - will be provided by user
    URLs = []
    
    # Get URLs from user input
    print("=" * 60)
    print("Web Scraper to Pinecone Vector Database")
    print("Using Sentence Transformers for Embeddings (No API Key Required!)")
    print("=" * 60)
    print("\nEnter URLs to scrape (one per line).")
    print("Type 'DONE' on a new line when finished, or press Enter twice:")
    print("-" * 60)
    
    while True:
        url = input().strip()
        if url.upper() == 'DONE' or (not url and URLs):
            break
        if url:
            URLs.append(url)
    
    if not URLs:
        print("\nNo URLs provided. Exiting.")
        return
    
    print(f"\n{len(URLs)} URL(s) to process:")
    for i, url in enumerate(URLs, 1):
        print(f"  {i}. {url}")
    
    # Ask about embeddings (now always available, no API key needed)
    use_embeddings = input("\nGenerate embeddings? [Y/n]: ").strip().lower()
    use_embeddings = use_embeddings != 'n'
    
    if use_embeddings:
        print(f"\nUsing embedding model: {EMBEDDING_MODEL}")
        print("(First run will download the model, subsequent runs will be faster)")
    
    # Initialize scraper
    scraper = WebScraperToPinecone(
        embedding_model=EMBEDDING_MODEL,
        chunk_size=2000,
        chunk_overlap=200
    )
    
    # Scrape URLs
    print(f"\nStarting to scrape {len(URLs)} URLs...")
    print("-" * 60)
    scraper.scrape_urls(URLs, generate_embeddings=use_embeddings)
    
    # Save to JSON file
    print("\n" + "-" * 60)
    scraper.save_to_pinecone_json(OUTPUT_FILE, NAMESPACE)
    
    print(f"\n{'=' * 60}")
    print(f"Done! Check {OUTPUT_FILE} for the Pinecone-compatible JSON file.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

