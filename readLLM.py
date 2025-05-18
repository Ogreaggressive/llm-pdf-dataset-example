import json
from typing import List, Dict
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pypdf
from docx import Document
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentHandler:
    """Base class for document handlers"""
    @staticmethod
    def can_handle(file_path: Path) -> bool:
        raise NotImplementedError
        
    @staticmethod
    def extract_text(file_path: Path) -> str:
        raise NotImplementedError

class PDFHandler(DocumentHandler):
    @staticmethod
    def can_handle(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'
        
    @staticmethod
    def extract_text(file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

class DOCXHandler(DocumentHandler):
    @staticmethod
    def can_handle(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.docx'
        
    @staticmethod
    def extract_text(file_path: Path) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

class TextHandler(DocumentHandler):
    @staticmethod
    def can_handle(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.txt'
        
    @staticmethod
    def extract_text(file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class DocumentDatasetGenerator:
    def __init__(self, model_path: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the dataset generator.
        
        Args:
            model_path: Path to the model
            chunk_size: Size of text chunks to process
            overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize document handlers
        self.handlers = [PDFHandler, DOCXHandler, TextHandler]
        
    def get_handler(self, file_path: Path) -> DocumentHandler:
        """Get appropriate handler for file type."""
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return handler
        raise ValueError(f"No handler available for file type: {file_path.suffix}")
        
    def chunk_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust chunk end to not break mid-sentence
            if end < len(text):
                end = text.rfind('.', start, end) + 1
                if end <= start:  # No period found
                    end = start + self.chunk_size
            
            chunk = text[start:end].strip()
            chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - self.overlap
            
        return chunks

    def generate_response(self, prompt: str) -> str:
        """Generate response using the model."""
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps create high-quality training datasets from documents."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,  # Add some randomness for variety
            top_p=0.95
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def create_prompt(self, document_name: str, chunk: str) -> str:
        """Create the prompt for the LLM."""
        return f"""You are an LLM Agent that knows all the documents from the company. Create an LLM dataset entry from the following document chunk. The dataset should include:
1. An instruction that describes what information to extract
2. An input providing context or a specific question
3. A response with relevant information from the text

Format your response exactly as follows:
INSTRUCTION: According to {document_name}, provide information about...
INPUT: [A relevant question about the content]
RESPONSE: [Information from the text that answers the question]

Document chunk:
{chunk}

Create 2-3 different instruction-input-response sets from this content. Separate each set with ###."""

    def process_chunk(self, document_name: str, chunk: str) -> List[Dict]:
        """Process a single chunk using the LLM."""
        prompt = self.create_prompt(document_name, chunk)
        
        try:
            # Get response from LLM
            response = self.generate_response(prompt)
            
            # Parse the response into separate examples
            examples = response.split('###')
            dataset_entries = []
            
            for example in examples:
                if not example.strip():
                    continue
                    
                try:
                    # Extract instruction, input, and response
                    instruction = example.split('INSTRUCTION:')[1].split('INPUT:')[0].strip()
                    input_text = example.split('INPUT:')[1].split('RESPONSE:')[0].strip()
                    response = example.split('RESPONSE:')[1].strip()
                    
                    dataset_entries.append({
                        'instruction': instruction,
                        'input': input_text,
                        'output': response
                    })
                except IndexError:
                    logger.warning(f"Failed to parse example: {example}")
                    continue
                    
            return dataset_entries
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return []

    def process_document(self, document_path: Path) -> List[Dict]:
        """Process an entire document and generate dataset entries."""
        logger.info(f"Processing document: {document_path}")
        
        # Get appropriate handler and extract text
        handler = self.get_handler(document_path)
        text = handler.extract_text(document_path)
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into chunks
        chunks = self.chunk_document(text)
        dataset_entries = []
        
        # Process each chunk
        for chunk in tqdm(chunks, desc="Processing chunks"):
            entries = self.process_chunk(document_path.name, chunk)
            dataset_entries.extend(entries)
            
        return dataset_entries

    def process_directory(self, directory_path: str, output_path: str):
        """Process all documents in a directory and save the dataset."""
        directory = Path(directory_path)
        all_entries = []
        
        # Process each supported document
        for file_path in directory.iterdir():
            if any(handler.can_handle(file_path) for handler in self.handlers):
                entries = self.process_document(file_path)
                all_entries.extend(entries)
            
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_entries, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total examples generated: {len(all_entries)}")

def main():
    # Initialize dataset generator with model path
    model_path = r"Qwen2.5-Coder-1.5B-Instruct"
    generator = DocumentDatasetGenerator(model_path)
    
    # Process documents
    generator.process_directory(
        directory_path=r"PDFs",
        output_path="llm_dataset.json"
    )

if __name__ == "__main__":
    main()