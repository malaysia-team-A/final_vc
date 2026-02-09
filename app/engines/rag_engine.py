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
        1. FAISS vector index (documents + DB Collections)
        2. MongoDB collections (Structured Data: Hostel Prices, Schedules, etc.)
        """
        results = []
        
        # 1. FAISS Semantic Search (Docs + Ingested DB Collections)
        if self.enabled and self.index is not None and self.index.ntotal > 0:
            try:
                query_vector = self.model.encode([query])
                D, I = self.index.search(np.array(query_vector).astype('float32'), k=n_results)
                
                for idx in I[0]:
                    if idx != -1 and idx < len(self.metadata):
                        meta = self.metadata[idx]
                        text = meta.get('text', '')
                        source = meta.get('source', 'Unknown')
                        
                        # Format based on source
                        if source.startswith("DB:"):
                            coll_name = source.replace("DB:", "")
                            results.append(f"[DB: {coll_name}]\n{text}")
                        else:
                            results.append(f"[Document: {source}]\n{text}")
                            
            except Exception as e:
                print(f"FAISS Search Error: {e}")
        
        # 2. MongoDB Structured Search (Keywords for Specific Data)
        # Note: FAQ/Q&A collections are now handled by Vector Search above.
        # We only keep keyword search for highly structured data (Tables, Schedules)
        try:
            from .db_engine import db_engine
            if db_engine.connected and db_engine.db is not None:
                query_lower = query.lower()
                
                # Hostel (Price List / Room Types) - Structured Data
                if any(k in query_lower for k in ['hostel', 'room', 'accommodation', 'dorm', '기숙사', '숙소', 'block']):
                    hostels = list(db_engine.db['Hostel'].find({}, {"_id": 0}).limit(5))
                    if hostels:
                        # Format as a concise list/table
                        hostel_info = "\n".join([f"- {h.get('room_type', 'Room')}: RM{h.get('rent_price', 'N/A')}/mo ({h.get('building', '')})" for h in hostels])
                        results.append(f"[Hostel Prices]\n{hostel_info}")
                
                # Facility (Hours / Location) - Structured Data
                if any(k in query_lower for k in ['facility', 'gym', 'library', 'cafeteria', '시설', '도서관', '체육관']):
                    facilities = list(db_engine.db['UCSI_FACILITY'].find({}, {"_id": 0}).limit(5))
                    if facilities:
                        fac_info = "\n".join([f"- {f.get('name', '')}: {f.get('location', '')} ({f.get('opening_hours', '')})" for f in facilities])
                        results.append(f"[Facility Info]\n{fac_info}")
                
                # Schedule queries (Dates) - Structured Data
                if any(k in query_lower for k in ['schedule', 'deadline', 'registration', 'semester', '일정', '학기', '등록']):
                    schedules = list(db_engine.db['USCI_SCHEDUAL'].find({}, {"_id": 0}).limit(5))
                    if schedules:
                        sched_info = "\n".join([f"- {s.get('event_name', '')}: {s.get('start_date', '')} ~ {s.get('end_date', '')}" for s in schedules])
                        results.append(f"[Academic Schedule]\n{sched_info}")
                
                # Programme/Fee queries (Tabular) - Structured Data
                if any(k in query_lower for k in ['fee', 'tuition', 'cost', 'program', 'course', 'major', '학비', '전공', '프로그램']):
                    majors = list(db_engine.db['UCSI_ MAJOR'].find({}, {"_id": 0}).limit(3))
                    if majors:
                        major_info = "\n".join([f"- {m.get('Programme', '')}: RM{m.get('Local Students Fees', 'N/A')} ({m.get('Course Duration', '')})" for m in majors])
                        results.append(f"[Programme Fees]\n{major_info}")
                
                # Staff queries (Directory) - Structured Data
                if any(k in query_lower for k in ['staff', 'professor', 'lecturer', 'contact', '교수', '연락처', '담당자']):
                    staff_data = list(db_engine.db['UCSI_STAFF'].find({}, {"_id": 0}).limit(2))
                    if staff_data:
                        for dept in staff_data:
                            members = dept.get('staff_members', [])[:3]
                            staff_info = "\n".join([f"- {m.get('name', '')}: {m.get('role', '')} ({m.get('email', '')})" for m in members])
                            results.append(f"[Staff Directory - {dept.get('major', '')}]\n{staff_info}")
                
        except Exception as e:
            print(f"MongoDB RAG Search Error: {e}")
        
        return "\n\n".join(results) if results else ""

    # [ADDED] RAG Upgrade: Ingest MongoDB Collection
    def ingest_collection(self, collection_name: str, fields: List[str]):
        """
        Ingest a MongoDB collection into the vector DB.
        collection_name: Name of the collection (e.g., 'UCSI_HOSTEL_FAQ')
        fields: List of fields to combine for embedding (e.g., ['question', 'answer'])
        """
        if not self.enabled: return False
        
        try:
            from .db_engine import db_engine
            if not db_engine.connected:
                print(f"RAG Ingest Skipped: DB not connected for {collection_name}")
                return False

            print(f"[RAG] Ingesting collection: {collection_name}...")
            documents = list(db_engine.db[collection_name].find({}, {"_id": 0}))
            
            if not documents:
                print(f"[RAG] No documents found in {collection_name}")
                return False
            
            new_texts = []
            new_metadata = []
            
            for doc in documents:
                # Combine fields into a single string
                parts = [str(doc.get(f, '')) for f in fields if doc.get(f)]
                combined_text = "\n".join(parts)
                
                if combined_text.strip():
                    new_texts.append(combined_text)
                    new_metadata.append({"text": combined_text, "source": f"DB:{collection_name}"})
            
            if not new_texts:
                return False

            # Check for duplicates (Simple check)
            # In a production system, we would use IDs, but here we just check text exact match to avoid simple dups on restart
            # For efficiency, we only add what's not already in metadata
            unique_texts = []
            unique_meta = []
            
            existing_texts = {m['text'] for m in self.metadata if m.get('source', '').startswith('DB:')}
            
            for i, text in enumerate(new_texts):
                if text not in existing_texts:
                    unique_texts.append(text)
                    unique_meta.append(new_metadata[i])
            
            if unique_texts:
                print(f"[RAG] Adding {len(unique_texts)} new items from {collection_name}")
                embeddings = self.model.encode(unique_texts)
                self.index.add(np.array(embeddings).astype('float32'))
                self.metadata.extend(unique_meta)
                self._save_index()
                return True
            else:
                print(f"[RAG] No new unique items to add from {collection_name}")
                return True

        except Exception as e:
            print(f"Error ingesting collection {collection_name}: {e}")
            return False

# Singleton
rag_engine = RAGEngine()

# [MODIFIED] Auto-ingest critical DB collections on startup
# Q&A 컬렉션은 question+answer 또는 query+answer 구조로, 정형 데이터는 설명 필드로 색인
if HAS_DEPENDENCIES:
    try:
        # Q&A Collections (벡터 검색에 적합)
        rag_engine.ingest_collection('UCSI_HOSTEL_FAQ', ['question', 'answer'])
        rag_engine.ingest_collection('LearnedQA', ['query', 'answer'])
        rag_engine.ingest_collection('BadQA', ['query', 'answer'])  # 피드백 기반 나쁜 예시
        
        # Structured Collections (텍스트 설명 필드가 있는 경우 색인)
        # 시설: 이름, 위치, 운영시간을 하나의 텍스트로 결합
        rag_engine.ingest_collection('UCSI_FACILITY', ['name', 'location', 'opening_hours', 'description'])
        # 교직원: 이름, 역할, 전공을 텍스트로 결합 (이메일은 검색용 X)
        # Note: UCSI_STAFF는 nested structure일 수 있으므로 별도 처리 필요 시 수정
        # 전공: 프로그램, 기간, 학비를 텍스트로 결합
        rag_engine.ingest_collection('UCSI_ MAJOR', ['Programme', 'Course Duration', 'Local Students Fees'])
        # 건물 정보
        rag_engine.ingest_collection('UCSI_University_Blocks_Data', ['block_name', 'description', 'facilities'])
        
    except Exception as e:
        print(f"Startup Ingest Warning: {e}")

