"""
RAG Engine - Lightweight FAISS Version
Uses FAISS for vector search and Sentence Transformers for embeddings.
"""
import os
# Suppress HuggingFace Warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import pickle
from typing import List, Dict, Optional
import numpy as np

# Conditional imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: RAG dependencies missing. Install faiss-cpu, sentence-transformers, PyPDF2")

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
INDEX_FILE = "data/knowledge_base/faiss_index.bin"
METADATA_FILE = "data/knowledge_base/faiss_metadata.pkl"

class RAGEngine:
    def __init__(self):
        self.index = None
        self.metadata = []  # List of dicts: [{"text": "...", "source": "..."}, ...]
        self.model = None
        self.enabled = HAS_DEPENDENCIES
        
        if self.enabled:
            try:
                # Load Model (MiniLM is fast and light)
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384  # MiniLM dimension
                
                # Load or Create Index
                self._load_index()
                
            except Exception as e:
                print(f"RAG Init Error: {e}")
                self.enabled = False

    def _load_index(self):
        """Load index from disk if exists"""
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
                with open(METADATA_FILE, 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def _save_index(self):
        """Save index to disk"""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)

    def ingest_file(self, file_path: str) -> bool:
        """
        Ingest a file (PDF or TXT) into the vector DB
        """
        if not self.enabled: return False
        
        try:
            filename = os.path.basename(file_path)
            ext = filename.split('.')[-1].lower()
            text = ""
            
            # Extract Text
            if ext == 'pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            elif ext == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == 'csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    text = "\n".join([",".join(row) for row in rows])
            else:
                return False
            
            if not text.strip(): return False
            
            # Chunking (Simple)
            chunk_size = 500
            overlap = 50
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                batch = text[i:i+chunk_size]
                if len(batch) > 50: # Ignore very small validation chunks
                    chunks.append(batch)
            
            if not chunks: return False

            # Embed and Add to FAISS
            embeddings = self.model.encode(chunks)
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Update Metadata
            for chunk in chunks:
                self.metadata.append({"text": chunk, "source": filename})
            
            # Save persistently
            self._save_index()
            return True
            
        except Exception as e:
            print(f"Error ingesting file {file_path}: {e}")
            return False

    def search(self, query: str, n_results=3) -> str:
        """
        Search for relevant context from multiple sources:
        1. FAISS vector index (documents)
        2. MongoDB collections (Hostel, Facility, FAQ, Schedule, Staff, Major)
        """
        results = []
        
        # 1. FAISS Document Search
        if self.enabled and self.index is not None and self.index.ntotal > 0:
            try:
                query_vector = self.model.encode([query])
                D, I = self.index.search(np.array(query_vector).astype('float32'), k=n_results)
                
                for idx in I[0]:
                    if idx != -1 and idx < len(self.metadata):
                        results.append(f"[Document] {self.metadata[idx]['text']}")
            except Exception as e:
                print(f"FAISS Search Error: {e}")
        
        # 2. MongoDB Collection Search
        try:
            from .db_engine import db_engine
            if db_engine.connected and db_engine.db:
                query_lower = query.lower()
                
                # Hostel queries
                if any(k in query_lower for k in ['hostel', 'room', 'accommodation', 'dorm', '기숙사', '숙소']):
                    hostels = list(db_engine.db['Hostel'].find({}, {"_id": 0}).limit(5))
                    if hostels:
                        hostel_info = "\n".join([f"- {h.get('room_type', 'Room')}: RM{h.get('rent_price', 'N/A')}/month, Deposit: RM{h.get('deposit', 'N/A')}, Building: {h.get('building', '')}" for h in hostels])
                        results.append(f"[Hostel Info]\n{hostel_info}")
                
                # Facility queries
                if any(k in query_lower for k in ['facility', 'gym', 'library', 'cafeteria', '시설', '도서관', '체육관']):
                    facilities = list(db_engine.db['UCSI_FACILITY'].find({}, {"_id": 0}).limit(5))
                    if facilities:
                        fac_info = "\n".join([f"- {f.get('name', '')}: {f.get('location', '')}, Hours: {f.get('opening_hours', '')}" for f in facilities])
                        results.append(f"[Facility Info]\n{fac_info}")
                
                # FAQ queries (Hostel)
                if any(k in query_lower for k in ['faq', 'question', 'how to', '질문', '방법']):
                    faqs = list(db_engine.db['UCSI_HOSTEL_FAQ'].find({}, {"_id": 0}).limit(3))
                    if faqs:
                        faq_info = "\n".join([f"Q: {f.get('question', '')}\nA: {f.get('answer', '')}" for f in faqs])
                        results.append(f"[FAQ]\n{faq_info}")
                
                # Schedule queries
                if any(k in query_lower for k in ['schedule', 'deadline', 'registration', 'semester', '일정', '학기', '등록']):
                    schedules = list(db_engine.db['USCI_SCHEDUAL'].find({}, {"_id": 0}).limit(5))
                    if schedules:
                        sched_info = "\n".join([f"- {s.get('event_name', '')}: {s.get('start_date', '')} ~ {s.get('end_date', '')}" for s in schedules])
                        results.append(f"[Academic Schedule]\n{sched_info}")
                
                # Programme/Fee queries
                if any(k in query_lower for k in ['fee', 'tuition', 'cost', 'program', 'course', 'major', '학비', '전공', '프로그램']):
                    majors = list(db_engine.db['UCSI_ MAJOR'].find({}, {"_id": 0}).limit(3))
                    if majors:
                        major_info = "\n".join([f"- {m.get('Programme', '')}: Local Fee: {m.get('Local Students Fees', 'N/A')}, Duration: {m.get('Course Duration', '')}" for m in majors])
                        results.append(f"[Programme Info]\n{major_info}")
                
                # Staff queries
                if any(k in query_lower for k in ['staff', 'professor', 'lecturer', 'contact', '교수', '연락처', '담당자']):
                    staff_data = list(db_engine.db['UCSI_STAFF'].find({}, {"_id": 0}).limit(2))
                    if staff_data:
                        for dept in staff_data:
                            members = dept.get('staff_members', [])[:3]
                            staff_info = "\n".join([f"- {m.get('name', '')}: {m.get('role', '')}, {m.get('email', '')}" for m in members])
                            results.append(f"[Staff - {dept.get('major', '')}]\n{staff_info}")
                
        except Exception as e:
            print(f"MongoDB RAG Search Error: {e}")
        
        return "\n\n".join(results) if results else ""

# Singleton
rag_engine = RAGEngine()

if __name__ == "__main__":
    if HAS_DEPENDENCIES:
        print("Dependencies found. Initializing FAISS RAG...")
    else:
        print("RAG dependencies missing.")
