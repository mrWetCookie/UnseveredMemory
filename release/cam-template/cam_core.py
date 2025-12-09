#!/usr/bin/env python3
"""
CAM (Continuous Architectural Memory) Core Library
Provides embedding, storage, and retrieval for semantic memory
"""

import fnmatch
import glob
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import google.generativeai as genai
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cosine as cosine_distance

# Version
CAM_VERSION = "1.7.2"

# Configuration
CAM_DIR = os.path.join(os.path.dirname(__file__))
VECTORS_DB = os.path.join(CAM_DIR, "vectors.db")
METADATA_DB = os.path.join(CAM_DIR, "metadata.db")
GRAPH_DB = os.path.join(CAM_DIR, "graph.db")
OPERATIONS_LOG = os.path.join(CAM_DIR, "operations.log")

# Phase 5: File Index Configuration
FILE_INDEX_DB = os.path.join(CAM_DIR, "file_index.db")

# Default ingest patterns by source type
DEFAULT_INGEST_PATTERNS = {
    "code": [
        "**/*.py", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx",
        "**/*.go", "**/*.rs", "**/*.java", "**/*.c", "**/*.cpp", "**/*.h",
        "**/*.rb", "**/*.php", "**/*.swift", "**/*.kt", "**/*.scala",
        "**/*.sh", "**/*.bash", "**/*.zsh"
    ],
    "docs": [
        "**/*.md", "**/*.mdx", "**/*.rst", "**/*.txt",
        ".ai/**/*"
    ],
    "config": [
        "**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml",
        "**/*.ini", "**/*.cfg", "**/*.conf",
        "**/Dockerfile", "**/docker-compose*.yml",
        "**/.env.example", "**/Makefile"
    ]
}

# Valid source types for database storage (matches CHECK constraint in vectors.db)
# The database schema only allows these types - others must be mapped
VALID_DB_SOURCE_TYPES = {'code', 'docs', 'operation', 'external', 'conversation'}

# Map conceptual types to valid database types
# 'config' files are semantically similar to 'code' for storage purposes
SOURCE_TYPE_MAPPING = {
    'config': 'code',  # Config files stored as 'code' type
}

# Patterns to always ignore
DEFAULT_IGNORE_PATTERNS = [
    # Dependencies
    "node_modules/**", "**/node_modules/**",
    "venv/**", "**/venv/**", ".venv/**",
    "vendor/**", "**/vendor/**",
    "__pycache__/**", "**/__pycache__/**",
    ".git/**", "**/.git/**",
    # Build artifacts
    "dist/**", "**/dist/**",
    "build/**", "**/build/**",
    ".next/**", "**/.next/**",
    "out/**", "**/out/**",
    "target/**", "**/target/**",
    # IDE/Editor
    ".idea/**", ".vscode/**", ".cursor/**",
    # Lock files (large, generated)
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock",
    # Binary/generated
    "*.min.js", "*.min.css", "*.map",
    "*.pyc", "*.pyo", "*.so", "*.dylib",
    "*.exe", "*.dll", "*.bin",
    # Media (not text)
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico",
    "*.mp3", "*.mp4", "*.wav", "*.avi",
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx",
    # CAM internal
    ".claude/cam/*.db", ".claude/cam/venv/**",
    ".claude/cam/*.log", ".claude/cam/.backup/**"
]

# Maximum file size for ingestion (100KB default)
MAX_INGEST_FILE_SIZE = 100 * 1024

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    EMBEDDING_MODEL = "models/embedding-001"
else:
    EMBEDDING_MODEL = None


@dataclass
class Embedding:
    """Represents a vector embedding with metadata"""

    id: str
    content: str
    embedding: Optional[np.ndarray]
    source_type: str  # 'code', 'docs', 'operation', 'external', 'conversation'
    source_file: Optional[str]
    created_at: str


@dataclass
class Annotation:
    """Represents metadata annotation for an embedding"""

    id: str
    embedding_id: str
    metadata: Dict
    tags: List[str]
    confidence: float


@dataclass
class Relationship:
    """Represents a graph relationship between entities"""

    source_id: str
    target_id: str
    relationship_type: str
    weight: float
    metadata: Optional[Dict]


@dataclass
class FileIndexEntry:
    """Represents a tracked file in the file index (Phase 5)"""

    file_path: str
    content_hash: str
    file_size: int
    last_ingested_at: str
    embedding_id: Optional[str]
    source_type: str


class CAM:
    """Continuous Architectural Memory system"""

    def __init__(self):
        """Initialize CAM with database connections"""
        self.vectors_conn = sqlite3.connect(VECTORS_DB)
        self.metadata_conn = sqlite3.connect(METADATA_DB)
        self.graph_conn = sqlite3.connect(GRAPH_DB)
        self.file_index_conn = sqlite3.connect(FILE_INDEX_DB)
        self._ensure_file_index_table()
        self._log("CAM initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all database connections"""
        self.vectors_conn.close()
        self.metadata_conn.close()
        self.graph_conn.close()
        self.file_index_conn.close()

    def _ensure_file_index_table(self):
        """Create file_index table if not exists (Phase 5)"""
        cursor = self.file_index_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                file_size INTEGER,
                last_ingested_at TIMESTAMP,
                embedding_id TEXT,
                source_type TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_content_hash ON file_index(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_last_ingested ON file_index(last_ingested_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_source_type ON file_index(source_type)")
        self.file_index_conn.commit()

    def _log(self, message: str):
        """Append to operations log"""
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(OPERATIONS_LOG, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # PHASE 5: File Index Methods (v1.5.0)
    # Smart tracking of files for incremental ingestion
    # =========================================================================

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file contents"""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _should_ignore_file(self, file_path: str, ignore_patterns: List[str] = None) -> bool:
        """Check if file matches any ignore pattern"""
        patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        rel_path = file_path

        for pattern in patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(os.path.basename(rel_path), pattern):
                return True
        return False

    def _detect_source_type(self, file_path: str) -> Optional[str]:
        """Detect source type based on file extension/path"""
        ext = os.path.splitext(file_path)[1].lower()
        basename = os.path.basename(file_path)

        # Docs detection
        if ext in ['.md', '.mdx', '.rst', '.txt']:
            return 'docs'
        if '.ai/' in file_path or '/.ai/' in file_path:
            return 'docs'

        # Code detection
        code_exts = ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
                     '.c', '.cpp', '.h', '.rb', '.php', '.swift', '.kt', '.scala',
                     '.sh', '.bash', '.zsh']
        if ext in code_exts:
            return 'code'

        # Config detection
        config_exts = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']
        config_names = ['Dockerfile', 'Makefile', '.env.example']
        if ext in config_exts or basename in config_names:
            return 'config'

        return None

    def check_file_status(self, file_path: str) -> Tuple[str, Optional[str]]:
        """
        Check if file is new, modified, or unchanged (Phase 5)

        Args:
            file_path: Path to file to check

        Returns:
            Tuple of (status, existing_hash) where status is 'new', 'modified', or 'unchanged'
        """
        if not os.path.exists(file_path):
            return ('missing', None)

        current_hash = self._compute_file_hash(file_path)

        cursor = self.file_index_conn.cursor()
        cursor.execute(
            "SELECT content_hash FROM file_index WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()

        if not row:
            return ('new', None)

        stored_hash = row[0]
        if stored_hash != current_hash:
            return ('modified', stored_hash)

        return ('unchanged', stored_hash)

    def update_file_index(
        self,
        file_path: str,
        embedding_id: str,
        source_type: str,
        content_hash: Optional[str] = None
    ) -> None:
        """
        Update file index after ingestion (Phase 5)

        Args:
            file_path: Path to ingested file
            embedding_id: ID of created embedding
            source_type: Type of source (code, docs, config)
            content_hash: Pre-computed hash (computed if not provided)
        """
        if content_hash is None:
            content_hash = self._compute_file_hash(file_path)

        file_size = os.path.getsize(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()

        cursor = self.file_index_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO file_index
            (file_path, content_hash, file_size, last_ingested_at, embedding_id, source_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_path, content_hash, file_size, timestamp, embedding_id, source_type))
        self.file_index_conn.commit()
        self._log(f"Updated file index: {file_path} -> {embedding_id}")

    def get_file_index_entry(self, file_path: str) -> Optional[FileIndexEntry]:
        """Get file index entry for a path"""
        cursor = self.file_index_conn.cursor()
        cursor.execute(
            "SELECT file_path, content_hash, file_size, last_ingested_at, embedding_id, source_type "
            "FROM file_index WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()
        if row:
            return FileIndexEntry(*row)
        return None

    def scan_directory(
        self,
        directory: str,
        source_type: Optional[str] = None,
        ignore_patterns: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Scan directory and categorize files by status (Phase 5)

        Args:
            directory: Directory to scan
            source_type: Optional filter by source type
            ignore_patterns: Patterns to ignore

        Returns:
            Dict with keys 'new', 'modified', 'unchanged', 'ignored' containing file lists
        """
        results = {'new': [], 'modified': [], 'unchanged': [], 'ignored': [], 'skipped': []}

        # Get all files recursively
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') or d in ['.ai', '.claude']]

            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, directory)

                # Check ignore patterns
                if self._should_ignore_file(rel_path, ignore_patterns):
                    results['ignored'].append(file_path)
                    continue

                # Check file size
                try:
                    if os.path.getsize(file_path) > MAX_INGEST_FILE_SIZE:
                        results['skipped'].append(file_path)
                        continue
                except OSError:
                    results['skipped'].append(file_path)
                    continue

                # Detect source type
                detected_type = self._detect_source_type(file_path)
                if detected_type is None:
                    results['ignored'].append(file_path)
                    continue

                # Filter by source type if specified
                if source_type and detected_type != source_type:
                    continue

                # Check status
                status, _ = self.check_file_status(file_path)
                if status in results:
                    results[status].append(file_path)

        return results

    def ingest_file(
        self,
        file_path: str,
        source_type: Optional[str] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Ingest a single file with smart change detection (Phase 5)

        Args:
            file_path: Path to file to ingest
            source_type: Source type (auto-detected if not provided)
            force: Force re-ingest even if unchanged

        Returns:
            Embedding ID if ingested, None if skipped
        """
        # Auto-detect source type
        if source_type is None:
            source_type = self._detect_source_type(file_path)
            if source_type is None:
                self._log(f"Cannot detect source type for: {file_path}")
                return None

        # Check if changed (unless forced)
        if not force:
            status, _ = self.check_file_status(file_path)
            if status == 'unchanged':
                return None
            if status == 'missing':
                self._log(f"File not found: {file_path}")
                return None

        # Read and ingest
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            self._log(f"Error reading file {file_path}: {e}")
            return None

        # Store embedding
        emb_id = self.store_embedding(
            content=content,
            source_type=source_type,
            source_file=file_path
        )

        # Add annotation
        self.annotate(
            embedding_id=emb_id,
            metadata={
                "file": file_path,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "file_size": os.path.getsize(file_path),
            },
            tags=[source_type, os.path.basename(file_path)],
        )

        # Update file index
        self.update_file_index(file_path, emb_id, source_type)

        return emb_id

    def ingest_directory(
        self,
        directory: str,
        source_type: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        Ingest all eligible files in directory (Phase 5)

        Args:
            directory: Directory to ingest
            source_type: Optional filter by source type
            force: Force re-ingest all files
            dry_run: Only report what would be ingested

        Returns:
            Dict with ingestion statistics
        """
        scan_results = self.scan_directory(directory, source_type)

        stats = {
            'scanned': sum(len(v) for v in scan_results.values()),
            'new': len(scan_results['new']),
            'modified': len(scan_results['modified']),
            'unchanged': len(scan_results['unchanged']),
            'ignored': len(scan_results['ignored']),
            'skipped': len(scan_results['skipped']),
            'ingested': 0,
            'failed': 0,
            'files': []
        }

        if dry_run:
            stats['dry_run'] = True
            stats['would_ingest'] = scan_results['new'] + scan_results['modified']
            return stats

        # Ingest new and modified files
        files_to_ingest = scan_results['new'] + scan_results['modified']
        if force:
            files_to_ingest += scan_results['unchanged']

        for file_path in files_to_ingest:
            try:
                emb_id = self.ingest_file(file_path, source_type, force=True)
                if emb_id:
                    stats['ingested'] += 1
                    stats['files'].append({'path': file_path, 'id': emb_id, 'status': 'success'})
                else:
                    stats['failed'] += 1
                    stats['files'].append({'path': file_path, 'id': None, 'status': 'failed'})
            except Exception as e:
                stats['failed'] += 1
                stats['files'].append({'path': file_path, 'id': None, 'status': f'error: {e}'})

        return stats

    def file_index_stats(self) -> Dict[str, any]:
        """Get file index statistics"""
        cursor = self.file_index_conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM file_index")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT source_type, COUNT(*) FROM file_index GROUP BY source_type")
        by_type = dict(cursor.fetchall())

        cursor.execute("SELECT SUM(file_size) FROM file_index")
        total_size = cursor.fetchone()[0] or 0

        return {
            'total_files': total,
            'by_type': by_type,
            'total_size_bytes': total_size,
            'total_size_human': f"{total_size / 1024:.1f} KB" if total_size < 1024*1024 else f"{total_size / (1024*1024):.1f} MB"
        }

    # =========================================================================
    # END PHASE 5: File Index Methods
    # =========================================================================

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding using Gemini

        Args:
            text: Text to embed

        Returns:
            numpy array of embeddings or None if API key not configured
        """
        if not EMBEDDING_MODEL:
            self._log("WARNING: GEMINI_API_KEY not set, skipping embedding generation")
            return None

        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL, content=text, task_type="retrieval_document"
            )
            embedding = np.array(result["embedding"], dtype=np.float32)
            self._log(f"Generated embedding (dim={len(embedding)})")
            return embedding
        except Exception as e:
            self._log(f"ERROR: Embedding generation failed: {str(e)}")
            return None

    def store_embedding(
        self,
        content: str,
        source_type: str,
        source_file: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        generate_embedding: bool = True,
    ) -> str:
        """
        Store content with optional embedding

        Args:
            content: Text content to store
            source_type: Type of source ('code', 'docs', 'operation', 'external', 'conversation')
            source_file: Optional source file path
            embedding: Optional pre-computed embedding
            generate_embedding: Whether to generate embedding if not provided

        Returns:
            ID of stored embedding
        """
        # Normalize source_type to valid database type
        # Maps 'config' -> 'code' and any unknown types to 'code'
        if source_type not in VALID_DB_SOURCE_TYPES:
            original_type = source_type
            source_type = SOURCE_TYPE_MAPPING.get(source_type, 'code')
            self._log(f"Normalized source_type: {original_type} -> {source_type}")

        emb_id = self._generate_id(content)

        # Generate embedding if requested and not provided
        if embedding is None and generate_embedding:
            embedding = self.embed(content)

        # Serialize embedding as bytes
        emb_blob = embedding.tobytes() if embedding is not None else None

        cursor = self.vectors_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings (id, content, embedding, source_type, source_file, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                emb_id,
                content,
                emb_blob,
                source_type,
                source_file,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.vectors_conn.commit()

        self._log(
            f"Stored embedding: {emb_id} (type={source_type}, file={source_file})"
        )
        return emb_id

    def annotate(
        self,
        embedding_id: str,
        metadata: Dict,
        tags: List[str],
        confidence: float = 1.0,
    ) -> str:
        """
        Add metadata annotation to an embedding

        Args:
            embedding_id: ID of the embedding to annotate
            metadata: Dictionary of metadata
            tags: List of searchable tags
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            ID of annotation
        """
        ann_id = self._generate_id(f"{embedding_id}:{json.dumps(metadata)}")

        cursor = self.metadata_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO annotations (id, embedding_id, metadata, tags, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ann_id,
                embedding_id,
                json.dumps(metadata),
                json.dumps(tags),
                confidence,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.metadata_conn.commit()

        self._log(f"Annotated {embedding_id} -> {ann_id} (tags={tags})")
        return ann_id

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Add a graph relationship

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship (e.g., 'depends_on', 'conflicts_with')
            weight: Relationship strength (0.0 to 1.0)
            metadata: Optional additional context
        """
        cursor = self.graph_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO relationships (source_id, target_id, relationship_type, weight, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                source_id,
                target_id,
                relationship_type,
                weight,
                json.dumps(metadata) if metadata else None,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.graph_conn.commit()

        self._log(
            f"Relationship: {source_id} --{relationship_type}--> {target_id} (weight={weight})"
        )

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query(
        self, query_text: str, top_k: int = 5, source_type_filter: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Semantic search using vector similarity

        Args:
            query_text: Query string
            top_k: Number of results to return
            source_type_filter: Optional filter by source type

        Returns:
            List of (id, content, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embed(query_text)

        if query_embedding is None:
            self._log("WARNING: Cannot perform semantic search without embeddings")
            return []

        # Retrieve all embeddings (with optional filter)
        cursor = self.vectors_conn.cursor()

        if source_type_filter:
            cursor.execute(
                """
                SELECT id, content, embedding FROM embeddings
                WHERE embedding IS NOT NULL AND source_type = ?
            """,
                (source_type_filter,),
            )
        else:
            cursor.execute("""
                SELECT id, content, embedding FROM embeddings
                WHERE embedding IS NOT NULL
            """)

        results = []
        for row in cursor.fetchall():
            emb_id, content, emb_blob = row

            # Deserialize embedding
            stored_embedding = np.frombuffer(emb_blob, dtype=np.float32)

            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, stored_embedding)
            results.append((emb_id, content, float(similarity)))

        # Sort by similarity and return top K
        results.sort(key=lambda x: x[2], reverse=True)

        self._log(
            f"Query: '{query_text[:50]}...' returned {len(results[:top_k])} results"
        )
        return results[:top_k]

    def get_metadata(self, embedding_id: str) -> Optional[Dict]:
        """Retrieve metadata for an embedding"""
        cursor = self.metadata_conn.cursor()
        cursor.execute(
            """
            SELECT metadata, tags, confidence FROM annotations
            WHERE embedding_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (embedding_id,),
        )

        row = cursor.fetchone()
        if row:
            metadata, tags, confidence = row
            return {
                "metadata": json.loads(metadata),
                "tags": json.loads(tags),
                "confidence": confidence,
            }
        return None

    def get_embedding(self, embedding_id: str) -> Optional[Dict]:
        """
        Retrieve full embedding content by ID

        Args:
            embedding_id: ID of the embedding to retrieve

        Returns:
            Dictionary with id, content, source_type, source_file, created_at, and metadata
            or None if not found
        """
        cursor = self.vectors_conn.cursor()
        cursor.execute(
            """
            SELECT id, content, source_type, source_file, created_at
            FROM embeddings
            WHERE id = ?
        """,
            (embedding_id,),
        )

        row = cursor.fetchone()
        if row:
            emb_id, content, source_type, source_file, created_at = row
            result = {
                "id": emb_id,
                "content": content,
                "source_type": source_type,
                "source_file": source_file,
                "created_at": created_at,
            }

            # Also fetch metadata if available
            metadata = self.get_metadata(embedding_id)
            if metadata:
                result["metadata"] = metadata

            return result
        return None

    # =========================================================================
    # PHASE 6: Session Memory System (v1.5.2)
    # Intelligent session tracking and retrieval
    # =========================================================================

    def get_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get list of recent sessions with metadata.

        Extracts unique sessions from annotations and aggregates their data.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries with id, project, timestamps, operation counts
        """
        cursor = self.metadata_conn.cursor()

        # Query for session summaries first (type = session_summary)
        cursor.execute("""
            SELECT
                embedding_id,
                metadata,
                tags,
                created_at
            FROM annotations
            WHERE json_extract(metadata, '$.type') = 'session_summary'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        session_summaries = {}
        for row in cursor.fetchall():
            emb_id, metadata_str, tags, created_at = row
            try:
                metadata = json.loads(metadata_str)
                session_id = metadata.get('session_id', 'unknown')
                if session_id not in session_summaries:
                    session_summaries[session_id] = {
                        'session_id': session_id,
                        'project': metadata.get('project', 'unknown'),
                        'end_time': metadata.get('end_time', created_at),
                        'operations': metadata.get('operations', {}),
                        'files_modified': metadata.get('files_modified', []),
                        'summary_available': True,
                        'embedding_id': emb_id
                    }
            except json.JSONDecodeError:
                continue

        # If no session summaries, extract sessions from operation annotations
        if not session_summaries:
            cursor.execute("""
                SELECT DISTINCT
                    json_extract(metadata, '$.session_id') as session_id,
                    json_extract(metadata, '$.project') as project,
                    MAX(json_extract(metadata, '$.created')) as last_activity,
                    COUNT(*) as op_count
                FROM annotations
                WHERE json_extract(metadata, '$.session_id') IS NOT NULL
                  AND json_extract(metadata, '$.session_id') != 'unknown'
                GROUP BY json_extract(metadata, '$.session_id')
                ORDER BY last_activity DESC
                LIMIT ?
            """, (limit,))

            for row in cursor.fetchall():
                session_id, project, last_activity, op_count = row
                if session_id and session_id not in session_summaries:
                    session_summaries[session_id] = {
                        'session_id': session_id,
                        'project': project or 'unknown',
                        'end_time': last_activity,
                        'operation_count': op_count,
                        'summary_available': False
                    }

        # Sort by end_time and return
        sessions = list(session_summaries.values())
        sessions.sort(key=lambda x: x.get('end_time', ''), reverse=True)

        self._log(f"Retrieved {len(sessions)} sessions")
        return sessions[:limit]

    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        Get detailed summary for a specific session.

        Args:
            session_id: Full or partial (8-char) session ID

        Returns:
            Session summary dictionary or None if not found
        """
        cursor = self.metadata_conn.cursor()

        # Try to find session summary by ID (support partial matching)
        # Note: annotations and embeddings are in separate DBs, so query separately
        cursor.execute("""
            SELECT
                embedding_id,
                metadata,
                tags
            FROM annotations
            WHERE json_extract(metadata, '$.type') = 'session_summary'
              AND json_extract(metadata, '$.session_id') LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (f"{session_id}%",))

        row = cursor.fetchone()
        if row:
            emb_id, metadata_str, tags = row
            try:
                metadata = json.loads(metadata_str)

                # Get content from vectors.db separately
                content = None
                vcursor = self.vectors_conn.cursor()
                vcursor.execute("SELECT content FROM embeddings WHERE id = ?", (emb_id,))
                vrow = vcursor.fetchone()
                if vrow:
                    content = vrow[0]

                return {
                    'session_id': metadata.get('session_id'),
                    'project': metadata.get('project'),
                    'start_time': metadata.get('start_time'),
                    'end_time': metadata.get('end_time'),
                    'operations': metadata.get('operations', {}),
                    'files_modified': metadata.get('files_modified', []),
                    'key_activities': metadata.get('key_activities', []),
                    'content': content,
                    'embedding_id': emb_id
                }
            except json.JSONDecodeError:
                pass

        # Fallback: aggregate from operation annotations
        cursor.execute("""
            SELECT
                json_extract(metadata, '$.title') as title,
                json_extract(metadata, '$.created') as created,
                json_extract(metadata, '$.project') as project
            FROM annotations
            WHERE tags LIKE ?
            ORDER BY created DESC
        """, (f"%session-{session_id[:8]}%",))

        operations = cursor.fetchall()
        if operations:
            # Extract unique files from operation titles
            files = set()
            op_types = {'Edit': 0, 'Read': 0, 'Bash': 0, 'Write': 0}

            for title, created, project in operations:
                if title and title.startswith('Op:'):
                    parts = title.split()
                    if len(parts) >= 2:
                        op_type = parts[1]
                        if op_type in op_types:
                            op_types[op_type] += 1
                        if len(parts) >= 3:
                            files.add(parts[2])

            return {
                'session_id': session_id[:8],
                'project': operations[0][2] if operations else 'unknown',
                'end_time': operations[0][1] if operations else None,
                'operations': op_types,
                'files_modified': list(files)[:20],  # Limit to 20 files
                'operation_count': len(operations),
                'summary_available': False,
                'content': f"Session {session_id[:8]} - {len(operations)} operations recorded"
            }

        return None

    def get_last_session(self) -> Optional[Dict]:
        """
        Get the most recent session summary.

        Returns:
            Most recent session summary or None if no sessions found
        """
        sessions = self.get_sessions(limit=1)
        if sessions:
            session = sessions[0]
            # Get full details if we have a session ID
            if 'session_id' in session:
                return self.get_session_summary(session['session_id'])
        return None

    def store_session_summary(
        self,
        session_id: str,
        project: str,
        operations: Dict[str, int],
        files_modified: List[str],
        key_activities: List[str] = None,
        start_time: str = None,
        end_time: str = None
    ) -> str:
        """
        Store a structured session summary for easy retrieval.

        Args:
            session_id: The session identifier
            project: Project name
            operations: Dict of operation counts {'Edit': 5, 'Read': 10, ...}
            files_modified: List of modified file paths
            key_activities: Optional list of key activities performed
            start_time: Session start timestamp
            end_time: Session end timestamp

        Returns:
            Embedding ID of stored summary
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc).isoformat()

        # Generate human-readable content for semantic search
        total_ops = sum(operations.values())
        files_list = '\n'.join(f'- {f}' for f in files_modified[:20])
        activities_list = '\n'.join(f'- {a}' for a in (key_activities or []))

        content = f"""## Session {session_id[:8]} - {project}
**Date**: {end_time[:10]}
**Operations**: {total_ops}

### Files Modified
{files_list if files_list else '- No files modified'}

### Key Activities
{activities_list if activities_list else '- Session activities not summarized'}

### Operations Breakdown
- Edit: {operations.get('Edit', 0)}
- Write: {operations.get('Write', 0)}
- Read: {operations.get('Read', 0)}
- Bash: {operations.get('Bash', 0)}
"""

        # Store the embedding
        emb_id = self.store_embedding(
            content=content,
            source_type='docs',  # Session summaries are documentation
            source_file=f"session:{session_id}"
        )

        # Create annotation with structured metadata
        metadata = {
            'type': 'session_summary',
            'session_id': session_id,
            'project': project,
            'start_time': start_time,
            'end_time': end_time,
            'operations': operations,
            'files_modified': files_modified[:50],  # Limit stored files
            'key_activities': key_activities or []
        }

        self.annotate(
            embedding_id=emb_id,
            metadata=metadata,
            tags=['session-summary', f'session-{session_id[:8]}', 'auto-generated'],
            confidence=1.0
        )

        self._log(f"Stored session summary: {session_id[:8]} ({total_ops} operations)")
        return emb_id

    def query_by_metadata_type(
        self,
        query_text: str,
        metadata_type: str,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Semantic search filtered by metadata type.

        Args:
            query_text: Query string
            metadata_type: Filter by type field in metadata (e.g., 'session_summary', 'ephemeral_note')
            top_k: Number of results

        Returns:
            List of (id, content, score) tuples
        """
        # Get embedding IDs that match the metadata type
        cursor = self.metadata_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT embedding_id
            FROM annotations
            WHERE json_extract(metadata, '$.type') = ?
        """, (metadata_type,))

        valid_ids = set(row[0] for row in cursor.fetchall())

        if not valid_ids:
            return []

        # Perform semantic search
        all_results = self.query(query_text, top_k=top_k * 3)  # Get more to filter

        # Filter to only matching metadata types
        filtered = [(id, content, score) for id, content, score in all_results if id in valid_ids]

        return filtered[:top_k]

    # =========================================================================
    # Ralph Wiggum Integration Methods (v1.6.0)
    # =========================================================================

    def get_ralph_loops(
        self,
        limit: int = 10,
        outcome: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent Ralph loop summaries.

        Args:
            limit: Maximum number of loops to return
            outcome: Filter by outcome ('success', 'max_iterations', 'cancelled')

        Returns:
            List of loop summary dictionaries
        """
        cursor = self.metadata_conn.cursor()

        query = """
            SELECT id, embedding_id, metadata, tags, created_at
            FROM annotations
            WHERE json_extract(metadata, '$.type') = 'ralph_loop_summary'
        """
        params = []

        if outcome:
            query += " AND json_extract(metadata, '$.outcome') = ?"
            params.append(outcome)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        results = []
        vcursor = self.vectors_conn.cursor()
        for row in cursor.fetchall():
            # Get content from vectors.db
            content = None
            emb_id = row[1]
            if emb_id:
                vcursor.execute("SELECT content FROM embeddings WHERE id = ?", (emb_id,))
                vrow = vcursor.fetchone()
                if vrow:
                    content = vrow[0]

            results.append({
                'id': row[0],
                'embedding_id': emb_id,
                'content': content,
                'metadata': json.loads(row[2]) if row[2] else {},
                'tags': row[3],
                'created_at': row[4]
            })

        return results

    def get_ralph_iterations(self, loop_id: str) -> List[Dict]:
        """
        Get all iterations for a specific Ralph loop.

        Args:
            loop_id: The unique loop identifier

        Returns:
            List of iteration dictionaries ordered by iteration number
        """
        cursor = self.metadata_conn.cursor()

        cursor.execute("""
            SELECT id, embedding_id, metadata, created_at
            FROM annotations
            WHERE json_extract(metadata, '$.type') = 'ralph_iteration'
              AND json_extract(metadata, '$.loop_id') = ?
            ORDER BY json_extract(metadata, '$.iteration') ASC
        """, (loop_id,))

        results = []
        vcursor = self.vectors_conn.cursor()
        for row in cursor.fetchall():
            # Get content from vectors.db
            content = None
            emb_id = row[1]
            if emb_id:
                vcursor.execute("SELECT content FROM embeddings WHERE id = ?", (emb_id,))
                vrow = vcursor.fetchone()
                if vrow:
                    content = vrow[0]

            results.append({
                'id': row[0],
                'embedding_id': emb_id,
                'content': content,
                'metadata': json.loads(row[2]) if row[2] else {},
                'created_at': row[3]
            })

        return results

    def query_ralph_patterns(
        self,
        task_description: str,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Semantic search across Ralph loop summaries for similar tasks.

        Args:
            task_description: Description of the task to find patterns for
            top_k: Number of results to return

        Returns:
            List of (id, content, score) tuples
        """
        return self.query_by_metadata_type(
            task_description,
            "ralph_loop_summary",
            top_k
        )

    def store_ralph_iteration(
        self,
        loop_id: str,
        iteration: int,
        outcome: str,
        project: str,
        changes: Optional[str] = None
    ) -> str:
        """
        Store a Ralph iteration summary.

        Args:
            loop_id: Unique loop identifier
            iteration: Iteration number
            outcome: 'continue', 'complete', 'max_reached', 'cancelled'
            project: Project name
            changes: Optional description of changes made

        Returns:
            Embedding ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        content = f"""Ralph Iteration {iteration}
Loop ID: {loop_id}
Project: {project}
Outcome: {outcome}
Timestamp: {timestamp}
Changes: {changes or 'none'}"""

        metadata = {
            "type": "ralph_iteration",
            "loop_id": loop_id,
            "iteration": iteration,
            "outcome": outcome,
            "project": project
        }

        return self.annotate(
            content=content,
            metadata=metadata,
            tags=f"ralph,iteration,loop-{loop_id}"
        )

    def store_ralph_loop_summary(
        self,
        loop_id: str,
        total_iterations: int,
        outcome: str,
        project: str,
        prompt: str,
        started_at: str,
        files_modified: Optional[List[str]] = None
    ) -> str:
        """
        Store a comprehensive Ralph loop summary.

        Args:
            loop_id: Unique loop identifier
            total_iterations: Total iterations completed
            outcome: 'success', 'max_iterations', 'cancelled'
            project: Project name
            prompt: Original prompt (truncated)
            started_at: ISO timestamp of loop start
            files_modified: List of files modified during loop

        Returns:
            Embedding ID
        """
        ended_at = datetime.now(timezone.utc).isoformat()

        # Truncate prompt for storage
        prompt_truncated = prompt[:1000]
        if len(prompt) > 1000:
            prompt_truncated += "..."

        files_str = ", ".join(files_modified[:20]) if files_modified else "unknown"

        content = f"""Ralph Loop Complete
Loop ID: {loop_id}
Project: {project}
Total Iterations: {total_iterations}
Outcome: {outcome}
Started: {started_at}
Ended: {ended_at}
Files Modified: {files_str}

Original Prompt:
{prompt_truncated}"""

        metadata = {
            "type": "ralph_loop_summary",
            "loop_id": loop_id,
            "iterations": total_iterations,
            "outcome": outcome,
            "project": project,
            "started_at": started_at,
            "ended_at": ended_at
        }

        return self.annotate(
            content=content,
            metadata=metadata,
            tags=f"ralph,loop-summary,{outcome},project-{project}"
        )

    def get_relationships(
        self,
        node_id: str,
        direction: str = "outgoing",
        relationship_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get relationships for a node

        Args:
            node_id: Node ID
            direction: 'outgoing', 'incoming', or 'both'
            relationship_type: Optional filter by type

        Returns:
            List of relationship dictionaries
        """
        cursor = self.graph_conn.cursor()

        conditions = []
        params = []

        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(node_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(node_id)
        else:  # both
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([node_id, node_id])

        if relationship_type:
            conditions.append("relationship_type = ?")
            params.append(relationship_type)

        where_clause = " AND ".join(conditions)

        cursor.execute(
            f"""
            SELECT source_id, target_id, relationship_type, weight, metadata
            FROM relationships
            WHERE {where_clause}
            ORDER BY weight DESC
        """,
            params,
        )

        relationships = []
        for row in cursor.fetchall():
            source, target, rel_type, weight, metadata = row
            relationships.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relationship_type": rel_type,
                    "weight": weight,
                    "metadata": json.loads(metadata) if metadata else None,
                }
            )

        return relationships

    def stats(self) -> Dict:
        """Get CAM statistics"""
        v_cursor = self.vectors_conn.cursor()
        m_cursor = self.metadata_conn.cursor()
        g_cursor = self.graph_conn.cursor()

        v_cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = v_cursor.fetchone()[0]

        v_cursor.execute("SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL")
        embedded_count = v_cursor.fetchone()[0]

        m_cursor.execute("SELECT COUNT(*) FROM annotations")
        total_annotations = m_cursor.fetchone()[0]

        g_cursor.execute("SELECT COUNT(*) FROM relationships")
        total_relationships = g_cursor.fetchone()[0]

        return {
            "total_embeddings": total_embeddings,
            "embedded_count": embedded_count,
            "total_annotations": total_annotations,
            "total_relationships": total_relationships,
        }


# ============================================================================
# EVALUATION FRAMEWORK (v1.2.0)
# ============================================================================


@dataclass
class EvaluationResult:
    """Result of an evaluation run"""

    metric_name: str
    score: float
    details: Dict
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class CAMEvaluator:
    """
    Evaluation framework for CAM accuracy metrics.

    Implements three categories of evaluation:
    1. Intrinsic: Embedding quality, retrieval accuracy, graph correctness
    2. Extrinsic: Operational utility, task success rates
    3. Benchmarks: DMR, LoCoMo-style standardized tests

    Research basis:
    - RAGAS framework (2024 EACL)
    - Zep temporal knowledge graph (Jan 2025)
    - A-Mem agentic memory (Feb 2025)
    - MTEB embedding benchmarks
    """

    def __init__(self, cam_instance: "CAM"):
        self.cam = cam_instance
        self.results_history: List[EvaluationResult] = []

    # =========================================================================
    # PHASE 1: INTRINSIC EVALUATION
    # =========================================================================

    def eval_embeddings(
        self,
        test_pairs: Optional[List[Tuple[str, str, float]]] = None,
        use_builtin_tests: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate embedding quality using Semantic Textual Similarity (STS).

        Measures how well CAM's cosine similarity correlates with human
        judgments of semantic similarity.

        Args:
            test_pairs: List of (text_a, text_b, human_score) tuples
                       human_score should be 0.0-1.0
            use_builtin_tests: If True and no test_pairs provided, use builtin tests

        Returns:
            EvaluationResult with Pearson correlation coefficient
        """
        from scipy.stats import pearsonr, spearmanr

        # Builtin semantic similarity test pairs (code/docs domain)
        builtin_tests = [
            # High similarity pairs (expected ~0.85-1.0)
            ("fix the navbar component", "repair navbar element", 0.90),
            ("install npm dependencies", "run npm install", 0.88),
            ("create database schema", "define database tables", 0.82),
            ("implement user authentication", "add user login system", 0.85),
            ("refactor the API endpoint", "restructure the API route", 0.87),
            # Medium similarity pairs (expected ~0.5-0.7)
            ("fix the navbar component", "update user settings", 0.35),
            ("install npm dependencies", "configure webpack", 0.55),
            ("create database schema", "write API documentation", 0.40),
            ("implement user authentication", "add dark mode toggle", 0.25),
            # Low similarity pairs (expected ~0.0-0.3)
            ("fix the navbar component", "quantum entanglement theory", 0.05),
            ("install npm dependencies", "medieval castle architecture", 0.02),
            ("create database schema", "cooking recipe for pasta", 0.03),
            ("implement user authentication", "botanical garden plants", 0.04),
        ]

        if test_pairs is None:
            if use_builtin_tests:
                test_pairs = builtin_tests
            else:
                return EvaluationResult(
                    metric_name="embedding_sts_correlation",
                    score=0.0,
                    details={"error": "No test pairs provided"},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

        cam_similarities = []
        human_similarities = []
        pair_results = []

        for text_a, text_b, human_score in test_pairs:
            # Generate embeddings
            emb_a = self.cam.embed(text_a)
            emb_b = self.cam.embed(text_b)

            if emb_a is None or emb_b is None:
                continue

            # Calculate CAM's similarity
            cam_score = float(self.cam.cosine_similarity(emb_a, emb_b))

            cam_similarities.append(cam_score)
            human_similarities.append(human_score)
            pair_results.append(
                {
                    "text_a": text_a[:50],
                    "text_b": text_b[:50],
                    "human_score": human_score,
                    "cam_score": round(cam_score, 4),
                    "delta": round(abs(cam_score - human_score), 4),
                }
            )

        if len(cam_similarities) < 3:
            return EvaluationResult(
                metric_name="embedding_sts_correlation",
                score=0.0,
                details={"error": "Not enough valid pairs for correlation"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Calculate correlations
        pearson_r, pearson_p = pearsonr(cam_similarities, human_similarities)
        spearman_r, spearman_p = spearmanr(cam_similarities, human_similarities)

        # Mean Absolute Error
        mae = np.mean(
            [abs(c - h) for c, h in zip(cam_similarities, human_similarities)]
        )

        result = EvaluationResult(
            metric_name="embedding_sts_correlation",
            score=float(pearson_r),
            details={
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 6),
                "spearman_r": round(float(spearman_r), 4),
                "spearman_p": round(float(spearman_p), 6),
                "mean_absolute_error": round(float(mae), 4),
                "n_pairs": len(cam_similarities),
                "pair_results": pair_results[:10],  # First 10 for brevity
                "interpretation": self._interpret_correlation(pearson_r),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    def eval_retrieval(
        self,
        test_queries: Optional[List[Dict]] = None,
        use_builtin_tests: bool = True,
        top_k: int = 5,
    ) -> EvaluationResult:
        """
        Evaluate retrieval accuracy using precision, recall, and nDCG.

        Based on RAGAS context precision/recall metrics.

        Args:
            test_queries: List of {"query": str, "relevant_ids": [str]} dicts
            use_builtin_tests: If True and no test_queries, use existing CAM data
            top_k: Number of results to retrieve

        Returns:
            EvaluationResult with precision, recall, nDCG@k
        """
        # If no test queries provided, generate from existing CAM data
        if test_queries is None and use_builtin_tests:
            test_queries = self._generate_retrieval_tests()

        if not test_queries:
            return EvaluationResult(
                metric_name="retrieval_accuracy",
                score=0.0,
                details={"error": "No test queries available"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        precisions = []
        recalls = []
        ndcgs = []
        query_results = []

        for test in test_queries:
            query = test["query"]
            relevant_ids = set(test.get("relevant_ids", []))

            if not relevant_ids:
                continue

            # Execute query
            results = self.cam.query(query, top_k=top_k)
            retrieved_ids = [r[0] for r in results]  # (id, content, score)
            retrieved_set = set(retrieved_ids)

            # Calculate metrics
            relevant_retrieved = len(retrieved_set & relevant_ids)

            # Precision: What fraction of retrieved items are relevant?
            precision = (
                relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
            )

            # Recall: What fraction of relevant items were retrieved?
            recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0

            # nDCG@k: Normalized Discounted Cumulative Gain
            ndcg = self._compute_ndcg(retrieved_ids, relevant_ids, k=top_k)

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

            query_results.append(
                {
                    "query": query[:50],
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "ndcg": round(ndcg, 4),
                    "retrieved": len(retrieved_ids),
                    "relevant": len(relevant_ids),
                    "hits": relevant_retrieved,
                }
            )

        if not precisions:
            return EvaluationResult(
                metric_name="retrieval_accuracy",
                score=0.0,
                details={"error": "No valid test queries with relevant IDs"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Aggregate scores
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))
        mean_ndcg = float(np.mean(ndcgs))

        # F1 score
        f1 = (
            2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
            if (mean_precision + mean_recall) > 0
            else 0.0
        )

        result = EvaluationResult(
            metric_name="retrieval_accuracy",
            score=mean_ndcg,  # nDCG as primary score
            details={
                "context_precision": round(mean_precision, 4),
                "context_recall": round(mean_recall, 4),
                f"ndcg@{top_k}": round(mean_ndcg, 4),
                "f1_score": round(f1, 4),
                "n_queries": len(precisions),
                "query_results": query_results[:10],
                "interpretation": self._interpret_retrieval(
                    mean_ndcg, mean_precision, mean_recall
                ),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    def eval_graph(self) -> EvaluationResult:
        """
        Evaluate knowledge graph quality.

        Metrics:
        - Edge coherence: Are connected nodes semantically related?
        - Temporal consistency: Do timestamps follow logical ordering?
        - Node coverage: Are all embeddings represented in graph?
        - Graph density: Edges per node (indicates connectivity)

        Returns:
            EvaluationResult with graph quality metrics
        """
        from scipy.spatial.distance import cosine

        # Load graph data
        g_cursor = self.cam.graph_conn.cursor()

        # Get all nodes
        g_cursor.execute("SELECT id, node_type, properties FROM nodes")
        nodes = {
            row[0]: {"type": row[1], "props": json.loads(row[2]) if row[2] else {}}
            for row in g_cursor.fetchall()
        }

        # Get all relationships
        g_cursor.execute("""
            SELECT source_id, target_id, relationship_type, weight, metadata
            FROM relationships
        """)
        edges = [
            {
                "source": row[0],
                "target": row[1],
                "type": row[2],
                "weight": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
            }
            for row in g_cursor.fetchall()
        ]

        if not nodes:
            return EvaluationResult(
                metric_name="graph_quality",
                score=0.0,
                details={
                    "error": "No nodes in graph",
                    "recommendation": "Run 'cam.sh graph build'",
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Metric 1: Edge coherence (semantic similarity between connected nodes)
        edge_coherences = []
        for edge in edges[:50]:  # Sample for performance
            source_props = nodes.get(edge["source"], {}).get("props", {})
            target_props = nodes.get(edge["target"], {}).get("props", {})

            source_ids = source_props.get("embedding_ids", [])
            target_ids = target_props.get("embedding_ids", [])

            if source_ids and target_ids:
                # Get centroids and compute similarity
                source_centroid = self._get_node_centroid(source_ids)
                target_centroid = self._get_node_centroid(target_ids)

                if source_centroid is not None and target_centroid is not None:
                    similarity = 1.0 - cosine(source_centroid, target_centroid)
                    edge_coherences.append(similarity)

        mean_edge_coherence = (
            float(np.mean(edge_coherences)) if edge_coherences else 0.0
        )

        # Metric 2: Temporal consistency
        temporal_violations = 0
        temporal_edges = [e for e in edges if e["type"] == "temporal"]
        for edge in temporal_edges:
            # For temporal edges, source should precede target
            source_time = self._get_node_latest_timestamp(nodes.get(edge["source"], {}))
            target_time = self._get_node_earliest_timestamp(
                nodes.get(edge["target"], {})
            )

            if source_time and target_time and source_time > target_time:
                temporal_violations += 1

        temporal_consistency = (
            1.0 - (temporal_violations / len(temporal_edges)) if temporal_edges else 1.0
        )

        # Metric 3: Node coverage (embeddings with graph representation)
        v_cursor = self.cam.vectors_conn.cursor()
        v_cursor.execute("SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL")
        total_embeddings = v_cursor.fetchone()[0]

        embeddings_in_graph = set()
        for node in nodes.values():
            emb_ids = node.get("props", {}).get("embedding_ids", [])
            embeddings_in_graph.update(emb_ids)

        node_coverage = (
            len(embeddings_in_graph) / total_embeddings if total_embeddings > 0 else 0.0
        )

        # Metric 4: Graph density
        num_nodes = len(nodes)
        num_edges = len(edges)
        max_edges = num_nodes * (num_nodes - 1)  # Directed graph
        density = num_edges / max_edges if max_edges > 0 else 0.0

        # Composite score (weighted average)
        composite_score = (
            0.4 * mean_edge_coherence
            + 0.2 * temporal_consistency
            + 0.3 * node_coverage
            + 0.1 * min(density * 10, 1.0)  # Cap density contribution
        )

        result = EvaluationResult(
            metric_name="graph_quality",
            score=float(composite_score),
            details={
                "edge_coherence": round(mean_edge_coherence, 4),
                "temporal_consistency": round(temporal_consistency, 4),
                "node_coverage": round(node_coverage, 4),
                "graph_density": round(density, 6),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "embeddings_in_graph": len(embeddings_in_graph),
                "total_embeddings": total_embeddings,
                "temporal_violations": temporal_violations,
                "edge_types": dict(Counter(e["type"] for e in edges)),
                "interpretation": self._interpret_graph(
                    composite_score, mean_edge_coherence, node_coverage
                ),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_ndcg(
        self, retrieved_ids: List[str], relevant_ids: set, k: int
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at k"""

        def dcg(relevances: List[int]) -> float:
            return sum(
                rel / np.log2(i + 2)  # +2 because i starts at 0
                for i, rel in enumerate(relevances[:k])
            )

        # Actual relevance scores (1 if relevant, 0 otherwise)
        actual_relevances = [1 if rid in relevant_ids else 0 for rid in retrieved_ids]

        # Ideal relevance scores (all relevant items first)
        ideal_relevances = sorted(actual_relevances, reverse=True)

        actual_dcg = dcg(actual_relevances)
        ideal_dcg = dcg(ideal_relevances)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def _generate_retrieval_tests(self) -> List[Dict]:
        """
        Generate retrieval test queries from existing CAM data.

        Strategy: Use content from existing embeddings as queries,
        mark original embedding as relevant.
        """
        cursor = self.cam.vectors_conn.cursor()
        cursor.execute("""
            SELECT id, content, source_type
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 20
        """)

        tests = []
        rows = cursor.fetchall()

        for emb_id, content, source_type in rows:
            # Use first 100 chars as query
            query = content[:100].replace("\n", " ")

            # Find semantically similar embeddings as "relevant"
            similar = self.cam.query(query, top_k=3)
            relevant_ids = [r[0] for r in similar if r[2] > 0.7]  # High similarity

            if relevant_ids:
                tests.append(
                    {
                        "query": query,
                        "relevant_ids": relevant_ids,
                        "source_type": source_type,
                    }
                )

        return tests

    def _get_node_centroid(self, embedding_ids: List[str]) -> Optional[np.ndarray]:
        """Get centroid (average embedding) for a set of embedding IDs"""
        if not embedding_ids:
            return None

        cursor = self.cam.vectors_conn.cursor()
        placeholders = ",".join("?" * len(embedding_ids))
        cursor.execute(
            f"""
            SELECT embedding
            FROM embeddings
            WHERE id IN ({placeholders})
            AND embedding IS NOT NULL
        """,
            embedding_ids,
        )

        embeddings = [
            np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()
        ]

        if not embeddings:
            return None

        return np.mean(embeddings, axis=0)

    def _get_node_latest_timestamp(self, node: Dict) -> Optional[datetime]:
        """Get latest timestamp from node's embeddings"""
        emb_ids = node.get("props", {}).get("embedding_ids", [])
        if not emb_ids:
            return None

        cursor = self.cam.vectors_conn.cursor()
        placeholders = ",".join("?" * len(emb_ids))
        cursor.execute(
            f"""
            SELECT MAX(created_at)
            FROM embeddings
            WHERE id IN ({placeholders})
        """,
            emb_ids,
        )

        row = cursor.fetchone()
        if row and row[0]:
            ts = row[0]
            if "Z" in ts or "+" in ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        return None

    def _get_node_earliest_timestamp(self, node: Dict) -> Optional[datetime]:
        """Get earliest timestamp from node's embeddings"""
        emb_ids = node.get("props", {}).get("embedding_ids", [])
        if not emb_ids:
            return None

        cursor = self.cam.vectors_conn.cursor()
        placeholders = ",".join("?" * len(emb_ids))
        cursor.execute(
            f"""
            SELECT MIN(created_at)
            FROM embeddings
            WHERE id IN ({placeholders})
        """,
            emb_ids,
        )

        row = cursor.fetchone()
        if row and row[0]:
            ts = row[0]
            if "Z" in ts or "+" in ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        return None

    def _interpret_correlation(self, r: float) -> str:
        """Interpret Pearson correlation coefficient"""
        if r >= 0.9:
            return "Excellent: CAM embeddings align very well with human judgment"
        elif r >= 0.7:
            return "Good: CAM embeddings show strong correlation with human judgment"
        elif r >= 0.5:
            return "Moderate: CAM embeddings have reasonable alignment"
        elif r >= 0.3:
            return (
                "Weak: CAM embeddings show limited correlation, consider model tuning"
            )
        else:
            return "Poor: CAM embeddings poorly aligned, investigate embedding model"

    def _interpret_retrieval(self, ndcg: float, precision: float, recall: float) -> str:
        """Interpret retrieval metrics"""
        if ndcg >= 0.8 and precision >= 0.7:
            return "Excellent: CAM retrieval is highly accurate and well-ranked"
        elif ndcg >= 0.6:
            return (
                "Good: CAM retrieval performs well, minor ranking improvements possible"
            )
        elif ndcg >= 0.4:
            return "Moderate: CAM retrieval works but has room for improvement"
        else:
            return "Poor: CAM retrieval needs attention, consider re-embedding or threshold tuning"

    def _interpret_graph(self, score: float, coherence: float, coverage: float) -> str:
        """Interpret graph quality metrics"""
        issues = []
        if coherence < 0.5:
            issues.append(
                "low edge coherence (connected nodes aren't semantically related)"
            )
        if coverage < 0.5:
            issues.append("low coverage (many embeddings not in graph)")

        if score >= 0.7:
            return "Excellent: Knowledge graph is well-structured and connected"
        elif score >= 0.5:
            if issues:
                return f"Good: Graph functional but has issues: {', '.join(issues)}"
            return "Good: Graph is reasonably well-structured"
        else:
            return f"Needs improvement: {', '.join(issues) if issues else 'Consider rebuilding graph'}"

    def run_all_intrinsic(self) -> Dict[str, EvaluationResult]:
        """Run all Phase 1 (intrinsic) evaluations"""
        print("🔬 Running Intrinsic Evaluation Suite...")
        print("=" * 60)

        results = {}

        print("\n[#] 1/3: Evaluating embedding quality (STS correlation)...")
        results["embeddings"] = self.eval_embeddings()
        print(f"   Score: {results['embeddings'].score:.4f}")
        print(f"   {results['embeddings'].details.get('interpretation', '')}")

        print("\n>>> 2/3: Evaluating retrieval accuracy...")
        results["retrieval"] = self.eval_retrieval()
        print(f"   nDCG@5: {results['retrieval'].score:.4f}")
        print(
            f"   Precision: {results['retrieval'].details.get('context_precision', 0):.4f}"
        )
        print(f"   Recall: {results['retrieval'].details.get('context_recall', 0):.4f}")

        print("\n[^]  3/3: Evaluating graph quality...")
        results["graph"] = self.eval_graph()
        print(f"   Score: {results['graph'].score:.4f}")
        print(
            f"   Edge coherence: {results['graph'].details.get('edge_coherence', 0):.4f}"
        )
        print(
            f"   Node coverage: {results['graph'].details.get('node_coverage', 0):.4f}"
        )

        print("\n" + "=" * 60)
        print("[v] Intrinsic Evaluation Complete")

        # Overall score
        overall = np.mean([r.score for r in results.values()])
        print(f"📈 Overall Intrinsic Score: {overall:.4f}")

        return results

    # =========================================================================
    # PHASE 2: EXTRINSIC EVALUATION
    # =========================================================================

    def eval_extrinsic(self) -> EvaluationResult:
        """
        Evaluate operational utility of CAM.

        Analyzes:
        1. Operation success rates from operations.log
        2. CAM query patterns (frequency, diversity)
        3. Annotation quality (tag coverage, metadata richness)
        4. Session activity patterns
        5. Knowledge utilization (are stored items being retrieved?)

        Returns:
            EvaluationResult with operational metrics
        """
        print("[#] Running Extrinsic Evaluation...")

        metrics = {
            "operations": self._analyze_operations_log(),
            "query_patterns": self._analyze_query_patterns(),
            "annotation_quality": self._analyze_annotation_quality(),
            "knowledge_utilization": self._analyze_knowledge_utilization(),
            "temporal_activity": self._analyze_temporal_activity(),
        }

        # Compute composite score
        scores = []

        # Operations success rate (weight: 0.3)
        op_score = metrics["operations"].get("success_rate", 0.0)
        scores.append(op_score * 0.3)

        # Query diversity (weight: 0.2)
        query_score = min(metrics["query_patterns"].get("unique_query_ratio", 0.0), 1.0)
        scores.append(query_score * 0.2)

        # Annotation quality (weight: 0.2)
        ann_score = metrics["annotation_quality"].get("quality_score", 0.0)
        scores.append(ann_score * 0.2)

        # Knowledge utilization (weight: 0.2)
        util_score = metrics["knowledge_utilization"].get("utilization_rate", 0.0)
        scores.append(util_score * 0.2)

        # Temporal activity (weight: 0.1)
        activity_score = min(
            metrics["temporal_activity"].get("activity_score", 0.0), 1.0
        )
        scores.append(activity_score * 0.1)

        composite_score = sum(scores)

        result = EvaluationResult(
            metric_name="extrinsic_utility",
            score=float(composite_score),
            details={
                "composite_score": round(composite_score, 4),
                "operations": metrics["operations"],
                "query_patterns": metrics["query_patterns"],
                "annotation_quality": metrics["annotation_quality"],
                "knowledge_utilization": metrics["knowledge_utilization"],
                "temporal_activity": metrics["temporal_activity"],
                "interpretation": self._interpret_extrinsic(composite_score, metrics),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    def _analyze_operations_log(self) -> Dict:
        """Analyze operations.log for success patterns"""
        log_path = os.path.join(CAM_DIR, "operations.log")

        if not os.path.exists(log_path):
            return {"error": "operations.log not found", "success_rate": 0.0}

        operations = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                operations.append(line)

        if not operations:
            return {"total_operations": 0, "success_rate": 0.0}

        # Parse operations
        success_count = 0
        failure_count = 0
        operation_types = Counter()

        for op in operations:
            # Look for success/failure indicators
            op_lower = op.lower()
            if "success: true" in op_lower or "success=true" in op_lower:
                success_count += 1
            elif (
                "success: false" in op_lower
                or "error" in op_lower
                or "failed" in op_lower
            ):
                failure_count += 1
            else:
                # Assume success if not explicitly failed
                success_count += 1

            # Extract operation type
            for op_type in ["Edit", "Write", "Bash", "Read", "Query", "Ingest"]:
                if op_type in op:
                    operation_types[op_type] += 1
                    break

        total = success_count + failure_count
        success_rate = success_count / total if total > 0 else 0.0

        return {
            "total_operations": len(operations),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_rate, 4),
            "operation_types": dict(operation_types),
            "recent_operations": len([o for o in operations[-100:]]),  # Last 100
        }

    def _analyze_query_patterns(self) -> Dict:
        """Analyze query diversity and patterns from stored embeddings"""
        cursor = self.cam.vectors_conn.cursor()

        # Get all operation-type embeddings (these often contain query info)
        cursor.execute("""
            SELECT content, created_at
            FROM embeddings
            WHERE source_type = 'operation'
            ORDER BY created_at DESC
            LIMIT 500
        """)

        rows = cursor.fetchall()

        if not rows:
            return {"total_queries": 0, "unique_query_ratio": 0.0}

        # Extract unique content patterns
        contents = [row[0][:100] for row in rows]  # First 100 chars
        unique_contents = set(contents)

        # Analyze query distribution
        word_counts = Counter()
        for content in contents:
            words = re.findall(r"\b\w{3,}\b", content.lower())
            word_counts.update(words)

        return {
            "total_queries": len(contents),
            "unique_queries": len(unique_contents),
            "unique_query_ratio": round(len(unique_contents) / len(contents), 4)
            if contents
            else 0.0,
            "top_terms": dict(word_counts.most_common(10)),
            "query_diversity_score": min(len(unique_contents) / 100, 1.0),  # Normalize
        }

    def _analyze_annotation_quality(self) -> Dict:
        """Analyze quality of metadata annotations"""
        cursor = self.cam.metadata_conn.cursor()

        cursor.execute("""
            SELECT metadata, tags, confidence
            FROM annotations
        """)

        rows = cursor.fetchall()

        if not rows:
            return {"total_annotations": 0, "quality_score": 0.0}

        # Analyze annotation richness
        total_annotations = len(rows)
        metadata_rich = 0  # Has multiple metadata fields
        tag_rich = 0  # Has multiple tags
        high_confidence = 0

        all_tags = Counter()

        for metadata_json, tags_json, confidence in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                tags = json.loads(tags_json) if tags_json else []

                if len(metadata) >= 2:
                    metadata_rich += 1

                if len(tags) >= 2:
                    tag_rich += 1

                if confidence and confidence >= 0.8:
                    high_confidence += 1

                all_tags.update(tags)

            except json.JSONDecodeError:
                continue

        # Quality score components
        metadata_ratio = metadata_rich / total_annotations if total_annotations else 0
        tag_ratio = tag_rich / total_annotations if total_annotations else 0
        confidence_ratio = (
            high_confidence / total_annotations if total_annotations else 0
        )

        quality_score = (
            (metadata_ratio * 0.4) + (tag_ratio * 0.4) + (confidence_ratio * 0.2)
        )

        return {
            "total_annotations": total_annotations,
            "metadata_rich_count": metadata_rich,
            "tag_rich_count": tag_rich,
            "high_confidence_count": high_confidence,
            "quality_score": round(quality_score, 4),
            "unique_tags": len(all_tags),
            "top_tags": dict(all_tags.most_common(10)),
        }

    def _analyze_knowledge_utilization(self) -> Dict:
        """Analyze how stored knowledge is being utilized"""
        v_cursor = self.cam.vectors_conn.cursor()

        # Get all embeddings with timestamps
        v_cursor.execute("""
            SELECT id, content, source_type, created_at
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
        """)

        rows = v_cursor.fetchall()

        if not rows:
            return {"total_embeddings": 0, "utilization_rate": 0.0}

        total_embeddings = len(rows)

        # Check which embeddings have been "utilized" (appear in relationships or annotations)
        utilized_ids = set()

        # Check graph relationships
        g_cursor = self.cam.graph_conn.cursor()
        g_cursor.execute("SELECT DISTINCT source_id FROM relationships")
        utilized_ids.update(row[0] for row in g_cursor.fetchall())
        g_cursor.execute("SELECT DISTINCT target_id FROM relationships")
        utilized_ids.update(row[0] for row in g_cursor.fetchall())

        # Check annotations
        m_cursor = self.cam.metadata_conn.cursor()
        m_cursor.execute("SELECT DISTINCT embedding_id FROM annotations")
        utilized_ids.update(row[0] for row in m_cursor.fetchall())

        # Calculate utilization
        embedding_ids = set(row[0] for row in rows)
        utilized_count = len(utilized_ids & embedding_ids)
        utilization_rate = (
            utilized_count / total_embeddings if total_embeddings else 0.0
        )

        # Analyze by source type
        source_type_counts = Counter(row[2] for row in rows)

        return {
            "total_embeddings": total_embeddings,
            "utilized_embeddings": utilized_count,
            "utilization_rate": round(utilization_rate, 4),
            "source_type_distribution": dict(source_type_counts),
            "recommendation": "Consider running 'cam.sh graph build' to increase utilization"
            if utilization_rate < 0.5
            else "Good utilization",
        }

    def _analyze_temporal_activity(self) -> Dict:
        """Analyze temporal patterns of CAM activity"""
        cursor = self.cam.vectors_conn.cursor()

        cursor.execute("""
            SELECT created_at
            FROM embeddings
            ORDER BY created_at DESC
            LIMIT 1000
        """)

        rows = cursor.fetchall()

        if not rows:
            return {"total_activity": 0, "activity_score": 0.0}

        # Parse timestamps
        timestamps = []
        for row in rows:
            ts_str = row[0]
            try:
                if "Z" in ts_str or "+" in ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                timestamps.append(ts)
            except (ValueError, TypeError):
                continue

        if not timestamps:
            return {"total_activity": 0, "activity_score": 0.0}

        # Activity metrics
        now = datetime.now(timezone.utc)
        last_24h = sum(1 for ts in timestamps if (now - ts).total_seconds() < 86400)
        last_7d = sum(1 for ts in timestamps if (now - ts).total_seconds() < 604800)
        last_30d = sum(1 for ts in timestamps if (now - ts).total_seconds() < 2592000)

        # Activity score based on recency
        if last_24h > 0:
            activity_score = 1.0
        elif last_7d > 0:
            activity_score = 0.7
        elif last_30d > 0:
            activity_score = 0.4
        else:
            activity_score = 0.1

        # Calculate daily average over last 30 days
        daily_avg = last_30d / 30 if last_30d else 0

        return {
            "total_activity": len(timestamps),
            "last_24h": last_24h,
            "last_7d": last_7d,
            "last_30d": last_30d,
            "daily_average": round(daily_avg, 2),
            "activity_score": round(activity_score, 4),
            "most_recent": timestamps[0].isoformat() if timestamps else None,
        }

    def _interpret_extrinsic(self, score: float, metrics: Dict) -> str:
        """Interpret extrinsic evaluation results"""
        issues = []

        ops = metrics.get("operations", {})
        if ops.get("success_rate", 1.0) < 0.8:
            issues.append("low operation success rate")

        util = metrics.get("knowledge_utilization", {})
        if util.get("utilization_rate", 1.0) < 0.5:
            issues.append("low knowledge utilization (run graph build)")

        activity = metrics.get("temporal_activity", {})
        if activity.get("last_24h", 0) == 0:
            issues.append("no recent activity")

        if score >= 0.7:
            return "Excellent: CAM is actively used and providing value"
        elif score >= 0.5:
            if issues:
                return f"Good: CAM functional but note: {', '.join(issues)}"
            return "Good: CAM is being utilized effectively"
        elif score >= 0.3:
            return f"Moderate: {', '.join(issues) if issues else 'Consider increasing CAM usage'}"
        else:
            return f"Low utilization: {', '.join(issues) if issues else 'CAM needs more active use'}"

    # =========================================================================
    # PHASE 3: BENCHMARK SUITE
    # =========================================================================

    def benchmark_dmr(self) -> EvaluationResult:
        """
        Deep Memory Retrieval (DMR) Benchmark.

        Inspired by Zep's evaluation methodology (Jan 2025, 94.8% accuracy).

        Tests CAM's ability to:
        1. Retrieve specific facts from stored memory
        2. Handle temporal queries (what happened before/after X)
        3. Cross-reference related information
        4. Maintain context over multiple retrievals

        Returns:
            EvaluationResult with DMR accuracy score
        """
        print("[DMR] Running Deep Memory Retrieval Benchmark...")
        print("   Based on Zep evaluation methodology (Jan 2025)")

        # Generate test cases from existing CAM data
        test_cases = self._generate_dmr_test_cases()

        if not test_cases:
            return EvaluationResult(
                metric_name="dmr_benchmark",
                score=0.0,
                details={
                    "error": "Not enough data for DMR benchmark",
                    "min_required": 10,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Run tests
        results = {
            "fact_retrieval": self._dmr_fact_retrieval(test_cases),
            "temporal_reasoning": self._dmr_temporal_reasoning(test_cases),
            "cross_reference": self._dmr_cross_reference(test_cases),
            "context_maintenance": self._dmr_context_maintenance(test_cases),
        }

        # Calculate overall DMR score (weighted)
        dmr_score = (
            results["fact_retrieval"]["accuracy"] * 0.35
            + results["temporal_reasoning"]["accuracy"] * 0.25
            + results["cross_reference"]["accuracy"] * 0.25
            + results["context_maintenance"]["accuracy"] * 0.15
        )

        result = EvaluationResult(
            metric_name="dmr_benchmark",
            score=float(dmr_score),
            details={
                "dmr_accuracy": round(dmr_score, 4),
                "fact_retrieval": results["fact_retrieval"],
                "temporal_reasoning": results["temporal_reasoning"],
                "cross_reference": results["cross_reference"],
                "context_maintenance": results["context_maintenance"],
                "test_cases_used": len(test_cases),
                "zep_reference": "94.8% (Zep benchmark)",
                "interpretation": self._interpret_dmr(dmr_score),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    def _generate_dmr_test_cases(self) -> List[Dict]:
        """Generate DMR test cases from existing CAM data"""
        cursor = self.cam.vectors_conn.cursor()

        # Get recent embeddings with content
        cursor.execute("""
            SELECT id, content, source_type, created_at
            FROM embeddings
            WHERE embedding IS NOT NULL
            AND length(content) > 50
            ORDER BY created_at DESC
            LIMIT 50
        """)

        rows = cursor.fetchall()

        test_cases = []
        for emb_id, content, source_type, created_at in rows:
            # Extract key terms for fact retrieval test
            words = re.findall(r"\b[A-Za-z]{4,}\b", content)
            if len(words) >= 3:
                test_cases.append(
                    {
                        "id": emb_id,
                        "content": content,
                        "source_type": source_type,
                        "created_at": created_at,
                        "key_terms": words[:5],
                    }
                )

        return test_cases

    def _dmr_fact_retrieval(self, test_cases: List[Dict]) -> Dict:
        """Test: Can CAM retrieve specific facts?"""
        correct = 0
        total = min(len(test_cases), 20)  # Limit for performance

        for case in test_cases[:total]:
            # Query using key terms
            query = " ".join(case["key_terms"][:3])
            results = self.cam.query(query, top_k=5)

            # Check if original embedding is in top 5
            retrieved_ids = [r[0] for r in results]
            if case["id"] in retrieved_ids:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "description": "Retrieve specific facts using key terms",
        }

    def _dmr_temporal_reasoning(self, test_cases: List[Dict]) -> Dict:
        """Test: Can CAM handle temporal queries?"""
        if len(test_cases) < 2:
            return {"accuracy": 0.0, "error": "Not enough data"}

        # Sort by timestamp
        sorted_cases = sorted(test_cases, key=lambda x: x["created_at"])

        correct = 0
        total = 0

        # Test: Query for "recent" content
        for i in range(min(10, len(sorted_cases) - 1)):
            recent_case = sorted_cases[-(i + 1)]
            older_case = sorted_cases[i]

            # Query with recent case terms
            query = " ".join(recent_case["key_terms"][:2])
            results = self.cam.query(query, top_k=3)

            if results:
                # Check if recent case ranks higher than older
                retrieved_ids = [r[0] for r in results]
                if recent_case["id"] in retrieved_ids:
                    recent_rank = retrieved_ids.index(recent_case["id"])
                    older_rank = (
                        retrieved_ids.index(older_case["id"])
                        if older_case["id"] in retrieved_ids
                        else 999
                    )

                    if recent_rank <= older_rank:
                        correct += 1
                    total += 1
                else:
                    total += 1

        accuracy = correct / total if total > 0 else 0.5  # Default 0.5 if can't test

        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "description": "Temporal ordering of retrieval results",
        }

    def _dmr_cross_reference(self, test_cases: List[Dict]) -> Dict:
        """Test: Can CAM find related content?"""
        correct = 0
        total = 0

        # Group by source_type
        by_type = {}
        for case in test_cases:
            st = case["source_type"]
            if st not in by_type:
                by_type[st] = []
            by_type[st].append(case)

        # Test: Can we find items of same type?
        for source_type, cases in by_type.items():
            if len(cases) >= 2:
                # Query with first case
                query = " ".join(cases[0]["key_terms"][:3])
                results = self.cam.query(query, top_k=5)

                # Check if any other case of same type is retrieved
                retrieved_ids = set(r[0] for r in results)
                same_type_ids = set(c["id"] for c in cases[1:])

                if retrieved_ids & same_type_ids:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "description": "Find related content by source type",
        }

    def _dmr_context_maintenance(self, test_cases: List[Dict]) -> Dict:
        """Test: Does retrieval remain consistent?"""
        if len(test_cases) < 5:
            return {"accuracy": 0.0, "error": "Not enough data"}

        consistent = 0
        total = min(10, len(test_cases))

        for case in test_cases[:total]:
            query = " ".join(case["key_terms"][:3])

            # Query twice
            results1 = self.cam.query(query, top_k=3)
            results2 = self.cam.query(query, top_k=3)

            # Check consistency
            ids1 = [r[0] for r in results1]
            ids2 = [r[0] for r in results2]

            if ids1 == ids2:
                consistent += 1

        accuracy = consistent / total if total > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "consistent": consistent,
            "total": total,
            "description": "Query result consistency (determinism)",
        }

    def _interpret_dmr(self, score: float) -> str:
        """Interpret DMR benchmark results"""
        if score >= 0.90:
            return "Excellent: Matches or exceeds Zep benchmark (94.8%)"
        elif score >= 0.80:
            return "Very Good: Strong memory retrieval capabilities"
        elif score >= 0.70:
            return "Good: Effective memory retrieval with room for improvement"
        elif score >= 0.50:
            return "Moderate: Basic retrieval working, consider tuning"
        else:
            return "Needs improvement: Review embedding quality and data volume"

    def benchmark_locomo(self) -> EvaluationResult:
        """
        LoCoMo-style Benchmark (Long-Context Memory).

        Inspired by A-Mem evaluation methodology (Feb 2025).
        Tests ability to handle long-range dependencies across sessions.

        Tests:
        1. Session continuity: Can we retrieve info from past sessions?
        2. Topic threading: Can we follow topic evolution over time?
        3. Knowledge synthesis: Can we combine information from multiple sources?
        4. Memory decay resistance: Are older memories still accessible?

        Returns:
            EvaluationResult with LoCoMo accuracy score
        """
        print("📚 Running LoCoMo-style Benchmark (Long-Context Memory)...")
        print("   Based on A-Mem evaluation methodology (Feb 2025)")

        # Generate test cases
        test_data = self._generate_locomo_test_data()

        if not test_data["embeddings"]:
            return EvaluationResult(
                metric_name="locomo_benchmark",
                score=0.0,
                details={"error": "Not enough data for LoCoMo benchmark"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Run tests
        results = {
            "session_continuity": self._locomo_session_continuity(test_data),
            "topic_threading": self._locomo_topic_threading(test_data),
            "knowledge_synthesis": self._locomo_knowledge_synthesis(test_data),
            "memory_decay_resistance": self._locomo_memory_decay(test_data),
        }

        # Calculate overall LoCoMo score
        locomo_score = (
            results["session_continuity"]["score"] * 0.30
            + results["topic_threading"]["score"] * 0.25
            + results["knowledge_synthesis"]["score"] * 0.25
            + results["memory_decay_resistance"]["score"] * 0.20
        )

        result = EvaluationResult(
            metric_name="locomo_benchmark",
            score=float(locomo_score),
            details={
                "locomo_accuracy": round(locomo_score, 4),
                "session_continuity": results["session_continuity"],
                "topic_threading": results["topic_threading"],
                "knowledge_synthesis": results["knowledge_synthesis"],
                "memory_decay_resistance": results["memory_decay_resistance"],
                "total_embeddings_tested": len(test_data["embeddings"]),
                "time_span_days": test_data.get("time_span_days", 0),
                "interpretation": self._interpret_locomo(locomo_score),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.results_history.append(result)
        return result

    def _generate_locomo_test_data(self) -> Dict:
        """Generate test data for LoCoMo benchmark"""
        cursor = self.cam.vectors_conn.cursor()

        cursor.execute("""
            SELECT id, content, source_type, source_file, created_at
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY created_at ASC
        """)

        rows = cursor.fetchall()

        if not rows:
            return {"embeddings": [], "time_span_days": 0}

        embeddings = []
        for row in rows:
            emb_id, content, source_type, source_file, created_at = row
            embeddings.append(
                {
                    "id": emb_id,
                    "content": content,
                    "source_type": source_type,
                    "source_file": source_file,
                    "created_at": created_at,
                }
            )

        # Calculate time span
        time_span_days = 0
        if len(embeddings) >= 2:
            try:
                first = embeddings[0]["created_at"]
                last = embeddings[-1]["created_at"]

                if "Z" in first or "+" in first:
                    first_dt = datetime.fromisoformat(first.replace("Z", "+00:00"))
                else:
                    first_dt = datetime.fromisoformat(first)

                if "Z" in last or "+" in last:
                    last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                else:
                    last_dt = datetime.fromisoformat(last)

                time_span_days = (last_dt - first_dt).days
            except (ValueError, TypeError):
                pass

        return {"embeddings": embeddings, "time_span_days": time_span_days}

    def _locomo_session_continuity(self, test_data: Dict) -> Dict:
        """Test: Can we retrieve information across different time periods?"""
        embeddings = test_data["embeddings"]

        if len(embeddings) < 10:
            return {"score": 0.5, "error": "Insufficient data"}

        # Split into time buckets
        n = len(embeddings)
        early = embeddings[: n // 3]
        middle = embeddings[n // 3 : 2 * n // 3]
        late = embeddings[2 * n // 3 :]

        retrieved_early = 0
        total_tests = min(5, len(late))

        # Test: Can recent queries retrieve early memories?
        for late_emb in late[:total_tests]:
            words = re.findall(r"\b[A-Za-z]{4,}\b", late_emb["content"])
            if not words:
                continue

            query = " ".join(words[:3])
            results = self.cam.query(query, top_k=10)
            retrieved_ids = set(r[0] for r in results)

            # Check if any early embedding was retrieved
            early_ids = set(e["id"] for e in early)
            if retrieved_ids & early_ids:
                retrieved_early += 1

        score = retrieved_early / total_tests if total_tests > 0 else 0.5

        return {
            "score": round(score, 4),
            "retrieved_early": retrieved_early,
            "total_tests": total_tests,
            "description": "Retrieve early memories from recent queries",
        }

    def _locomo_topic_threading(self, test_data: Dict) -> Dict:
        """Test: Can we follow topic evolution?"""
        embeddings = test_data["embeddings"]

        # Group by source_type to track topic threads
        by_type = {}
        for emb in embeddings:
            st = emb["source_type"]
            if st not in by_type:
                by_type[st] = []
            by_type[st].append(emb)

        # Test: Query first item, can we find later items of same type?
        successful_threads = 0
        total_threads = 0

        for source_type, type_embs in by_type.items():
            if len(type_embs) >= 3:
                # Query with first embedding's content
                first_words = re.findall(r"\b[A-Za-z]{4,}\b", type_embs[0]["content"])
                if not first_words:
                    continue

                query = " ".join(first_words[:3])
                results = self.cam.query(query, top_k=10)
                retrieved_ids = set(r[0] for r in results)

                # Check if later embeddings in thread were found
                later_ids = set(e["id"] for e in type_embs[1:])
                if retrieved_ids & later_ids:
                    successful_threads += 1
                total_threads += 1

        score = successful_threads / total_threads if total_threads > 0 else 0.5

        return {
            "score": round(score, 4),
            "successful_threads": successful_threads,
            "total_threads": total_threads,
            "description": "Follow topic evolution across time",
        }

    def _locomo_knowledge_synthesis(self, test_data: Dict) -> Dict:
        """Test: Can we combine information from multiple sources?"""
        embeddings = test_data["embeddings"]

        # Count unique source types
        source_types = set(e["source_type"] for e in embeddings)

        if len(source_types) < 2:
            return {"score": 0.5, "note": "Single source type"}

        # Test: Query that should retrieve from multiple types
        successful_synthesis = 0
        total_tests = 0

        # Group some embeddings by random queries
        for i in range(min(5, len(embeddings) // 2)):
            # Take random embedding and query
            test_emb = embeddings[i * 2] if i * 2 < len(embeddings) else embeddings[0]
            words = re.findall(r"\b[A-Za-z]{4,}\b", test_emb["content"])
            if not words:
                continue

            query = " ".join(words[:2])  # Short query to be more general
            results = self.cam.query(query, top_k=10)

            # Check if results span multiple source types
            result_types = set()
            for result in results:
                # Find source type for this result
                for emb in embeddings:
                    if emb["id"] == result[0]:
                        result_types.add(emb["source_type"])
                        break

            if len(result_types) >= 2:
                successful_synthesis += 1
            total_tests += 1

        score = successful_synthesis / total_tests if total_tests > 0 else 0.5

        return {
            "score": round(score, 4),
            "successful_synthesis": successful_synthesis,
            "total_tests": total_tests,
            "source_types_available": len(source_types),
            "description": "Combine info from multiple source types",
        }

    def _locomo_memory_decay(self, test_data: Dict) -> Dict:
        """Test: Are older memories still accessible?"""
        embeddings = test_data["embeddings"]

        if len(embeddings) < 5:
            return {"score": 0.5, "error": "Insufficient data"}

        # Get oldest embeddings
        oldest = embeddings[:5]

        accessible = 0
        for old_emb in oldest:
            words = re.findall(r"\b[A-Za-z]{4,}\b", old_emb["content"])
            if not words:
                continue

            query = " ".join(words[:3])
            results = self.cam.query(query, top_k=5)
            retrieved_ids = [r[0] for r in results]

            if old_emb["id"] in retrieved_ids:
                accessible += 1

        score = accessible / len(oldest) if oldest else 0.5

        return {
            "score": round(score, 4),
            "accessible": accessible,
            "tested": len(oldest),
            "description": "Oldest memories still retrievable",
        }

    def _interpret_locomo(self, score: float) -> str:
        """Interpret LoCoMo benchmark results"""
        if score >= 0.80:
            return "Excellent: Strong long-context memory capabilities"
        elif score >= 0.65:
            return "Good: Effective memory across sessions"
        elif score >= 0.50:
            return "Moderate: Basic long-term memory working"
        else:
            return (
                "Needs improvement: Consider increasing data volume or rebuilding graph"
            )

    def run_all_benchmarks(self) -> Dict[str, EvaluationResult]:
        """Run all Phase 3 benchmarks"""
        print("🏆 Running Full Benchmark Suite...")
        print("=" * 60)

        results = {}

        print("\n")
        results["dmr"] = self.benchmark_dmr()
        print(f"   DMR Score: {results['dmr'].score:.4f}")

        print("\n")
        results["locomo"] = self.benchmark_locomo()
        print(f"   LoCoMo Score: {results['locomo'].score:.4f}")

        print("\n" + "=" * 60)
        print("[v] Benchmark Suite Complete")

        # Overall benchmark score
        overall = np.mean([r.score for r in results.values()])
        print(f"📈 Overall Benchmark Score: {overall:.4f}")

        return results

    def run_full_evaluation(self) -> Dict[str, any]:
        """
        Run complete evaluation: intrinsic + extrinsic + benchmarks.

        Returns comprehensive report of CAM system health.
        """
        print("🔬 Running Full CAM Evaluation Suite...")
        print("=" * 70)

        all_results = {}

        # Phase 1: Intrinsic
        print("\n[#] PHASE 1: INTRINSIC EVALUATION")
        print("-" * 40)
        intrinsic = self.run_all_intrinsic()
        all_results["intrinsic"] = {k: v.to_dict() for k, v in intrinsic.items()}

        # Phase 2: Extrinsic
        print("\n[#] PHASE 2: EXTRINSIC EVALUATION")
        print("-" * 40)
        extrinsic = self.eval_extrinsic()
        all_results["extrinsic"] = extrinsic.to_dict()

        # Phase 3: Benchmarks
        print("\n[#] PHASE 3: BENCHMARKS")
        print("-" * 40)
        benchmarks = self.run_all_benchmarks()
        all_results["benchmarks"] = {k: v.to_dict() for k, v in benchmarks.items()}

        # Overall Summary
        intrinsic_avg = np.mean([r.score for r in intrinsic.values()])
        extrinsic_score = extrinsic.score
        benchmark_avg = np.mean([r.score for r in benchmarks.values()])

        overall_score = (
            (intrinsic_avg * 0.4) + (extrinsic_score * 0.3) + (benchmark_avg * 0.3)
        )

        all_results["summary"] = {
            "intrinsic_average": round(intrinsic_avg, 4),
            "extrinsic_score": round(extrinsic_score, 4),
            "benchmark_average": round(benchmark_avg, 4),
            "overall_score": round(overall_score, 4),
            "grade": self._score_to_grade(overall_score),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print("\n" + "=" * 70)
        print("🏆 FULL EVALUATION COMPLETE")
        print("=" * 70)
        print(f"   Intrinsic Average:  {intrinsic_avg:.4f}")
        print(f"   Extrinsic Score:    {extrinsic_score:.4f}")
        print(f"   Benchmark Average:  {benchmark_avg:.4f}")
        print(f"   ─────────────────────────────")
        print(
            f"   OVERALL SCORE:      {overall_score:.4f} ({self._score_to_grade(overall_score)})"
        )
        print("=" * 70)

        return all_results

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.90:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.80:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.70:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.60:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.50:
            return "C-"
        elif score >= 0.40:
            return "D"
        else:
            return "F"


# ============================================================================
# GRAPH BUILDING FUNCTIONS (v1.1.0)
# ============================================================================


def load_embeddings_for_clustering(
    db_path: str, limit: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from vectors.db for clustering.

    Args:
        db_path: Path to vectors.db
        limit: Optional limit on number of embeddings to load

    Returns:
        (embedding_ids, embedding_vectors)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if limit:
        cursor.execute(
            """
            SELECT id, embedding
            FROM embeddings
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )
    else:
        cursor.execute("""
            SELECT id, embedding
            FROM embeddings
            WHERE embedding IS NOT NULL
        """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], np.array([])

    embedding_ids = [row[0] for row in rows]
    # Embeddings stored as BLOB (numpy array bytes)
    embedding_vectors = np.array(
        [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    )

    return embedding_ids, embedding_vectors


def cluster_embeddings_hierarchical(
    embedding_vectors: np.ndarray,
    threshold: float = 0.25,  # Distance threshold (lower = more similar)
) -> np.ndarray:
    """
    Cluster embeddings using hierarchical clustering.

    Args:
        embedding_vectors: (n_embeddings, embedding_dim) array
        threshold: Distance threshold for clustering

    Returns:
        Array of cluster labels for each embedding
    """
    if len(embedding_vectors) < 2:
        return np.zeros(len(embedding_vectors), dtype=int)

    # Compute linkage matrix (average linkage with cosine metric)
    linkage_matrix = linkage(embedding_vectors, method="average", metric="cosine")

    # Cut dendrogram at threshold
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    return cluster_labels


def extract_concept_name_from_cluster(
    embedding_ids: List[str], vectors_db_path: str
) -> str:
    """
    Extract a semantic concept name from cluster embeddings.
    Looks at embedding content and finds most common meaningful terms.

    Args:
        embedding_ids: List of embedding IDs in this cluster
        vectors_db_path: Path to vectors.db (contains content)

    Returns:
        Concept name (e.g., "cam_operations")
    """
    conn = sqlite3.connect(vectors_db_path)
    cursor = conn.cursor()

    # Fetch content for these embeddings (content is in vectors.db)
    placeholders = ",".join("?" * len(embedding_ids))
    cursor.execute(
        f"""
        SELECT content
        FROM embeddings
        WHERE id IN ({placeholders})
    """,
        embedding_ids,
    )

    contents = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not contents:
        return "unknown_concept"

    # Extract keywords (simple approach: find most common nouns/verbs)
    all_words = []
    for content in contents:
        # Extract words (alphanumeric, 3+ chars)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        all_words.extend(words)

    # Filter stopwords (basic list)
    stopwords = {
        "the",
        "and",
        "for",
        "are",
        "with",
        "this",
        "that",
        "from",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "will",
        "can",
        "not",
        "but",
        "all",
        "use",
        "using",
        "used",
    }
    meaningful_words = [w for w in all_words if w not in stopwords]

    if not meaningful_words:
        return "general_operations"

    # Get most common word
    word_counts = Counter(meaningful_words)
    top_word = word_counts.most_common(1)[0][0]

    # Make it a concept name (e.g., "cam" -> "cam_operations")
    concept_name = f"{top_word}_operations"

    return concept_name


def cluster_embeddings(
    vectors_db_path: str, metadata_db_path: str, limit: Optional[int] = None
) -> Dict[int, Dict]:
    """
    Main clustering function.

    Args:
        vectors_db_path: Path to vectors.db
        metadata_db_path: Path to metadata.db
        limit: Optional limit on number of embeddings

    Returns:
        {
            cluster_id: {
                'embedding_ids': [id1, id2, ...],
                'concept_name': 'navbar_component',
                'size': 5
            },
            ...
        }
    """
    print(f"[#] Loading embeddings from {vectors_db_path}")
    embedding_ids, embedding_vectors = load_embeddings_for_clustering(
        vectors_db_path, limit=limit
    )

    if len(embedding_vectors) == 0:
        print("[!]  No embeddings found")
        return {}

    print(f"[v] Loaded {len(embedding_ids)} embeddings")

    print(">>> Clustering embeddings...")
    cluster_labels = cluster_embeddings_hierarchical(embedding_vectors)

    # Group embeddings by cluster
    clusters = {}
    for embedding_id, cluster_id in zip(embedding_ids, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(embedding_id)

    print(f"[v] Found {len(clusters)} clusters")

    # Extract concept names
    print("[#]  Extracting concept names...")
    cluster_data = {}
    for cluster_id, emb_ids in clusters.items():
        concept_name = extract_concept_name_from_cluster(emb_ids, vectors_db_path)
        cluster_data[cluster_id] = {
            "embedding_ids": emb_ids,
            "concept_name": concept_name,
            "size": len(emb_ids),
        }
        print(f"  - Cluster {cluster_id}: {concept_name} ({len(emb_ids)} embeddings)")

    return cluster_data


def clear_graph_nodes(graph_db_path: str) -> int:
    """
    Clear all nodes from graph.db (for rebuilding).

    Args:
        graph_db_path: Path to graph.db

    Returns:
        Number of nodes deleted
    """
    conn = sqlite3.connect(graph_db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM nodes")
    deleted = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted


def create_graph_nodes(
    cluster_data: Dict[int, Dict],
    vectors_db_path: str,
    metadata_db_path: str,
    graph_db_path: str,
) -> Dict[int, str]:
    """
    Create graph nodes from cluster data and link embeddings to nodes.

    Creates a 'concept' node for each cluster and updates embedding
    annotations to reference their parent node.

    Args:
        cluster_data: Output from cluster_embeddings()
        vectors_db_path: Path to vectors.db (not used currently)
        metadata_db_path: Path to metadata.db
        graph_db_path: Path to graph.db

    Returns:
        {cluster_id: node_id} mapping
    """
    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()
    m_conn = sqlite3.connect(metadata_db_path)
    m_cursor = m_conn.cursor()

    node_mapping = {}

    print("[^]  Creating graph nodes...")

    for cluster_id, cluster_info in cluster_data.items():
        # Generate unique node ID
        concept_name = cluster_info["concept_name"]
        node_id = f"concept_{concept_name}_{cluster_id}"

        # Create node properties (stored as JSON)
        # Convert numpy types to native Python types for JSON serialization
        properties = {
            "concept_name": concept_name,
            "cluster_id": int(cluster_id),  # Convert numpy int32 to Python int
            "size": cluster_info["size"],
            "embedding_ids": cluster_info["embedding_ids"],
        }

        # Insert node into graph.db
        g_cursor.execute(
            """
            INSERT OR REPLACE INTO nodes (id, node_type, properties)
            VALUES (?, ?, ?)
        """,
            (node_id, "concept", json.dumps(properties)),
        )

        # Link embeddings to this node via metadata annotations
        for emb_id in cluster_info["embedding_ids"]:
            # Check if annotation exists for this embedding
            m_cursor.execute(
                """
                SELECT COUNT(*) FROM annotations
                WHERE embedding_id = ?
            """,
                (emb_id,),
            )

            annotation_exists = m_cursor.fetchone()[0] > 0

            if annotation_exists:
                # Update existing annotation with graph_node_id
                m_cursor.execute(
                    """
                    UPDATE annotations
                    SET metadata = json_set(
                        COALESCE(metadata, '{}'),
                        '$.graph_node_id',
                        ?
                    ),
                    updated_at = CURRENT_TIMESTAMP
                    WHERE embedding_id = ?
                """,
                    (node_id, emb_id),
                )
            else:
                # Create annotation with graph_node_id
                m_cursor.execute(
                    """
                    INSERT INTO annotations (
                        id, embedding_id, metadata, tags, confidence
                    )
                    VALUES (?, ?, ?, '[]', 1.0)
                """,
                    (f"ann_{emb_id}", emb_id, json.dumps({"graph_node_id": node_id})),
                )

        node_mapping[cluster_id] = node_id
        print(f"  [v] {node_id} ({cluster_info['size']} embeddings)")

    g_conn.commit()
    m_conn.commit()
    g_conn.close()
    m_conn.close()

    print(f"[v] Created {len(node_mapping)} graph nodes")

    return node_mapping


def discover_temporal_relationships(
    graph_db_path: str,
    metadata_db_path: str,
    vectors_db_path: str,
    time_window_hours: float = 168.0,  # 1 week default
) -> List[Tuple[str, str, float]]:
    """
    Discover temporal relationships between nodes based on embedding timestamps.

    Finds sequence patterns where embeddings in one node chronologically
    precede embeddings in another node, indicating temporal succession.

    Args:
        graph_db_path: Path to graph.db
        metadata_db_path: Path to metadata.db
        vectors_db_path: Path to vectors.db
        time_window_hours: Maximum time gap to consider (default 1 week)

    Returns:
        List of (source_node_id, target_node_id, weight) tuples
        Weight is based on temporal proximity (0-1, higher = more recent)
    """
    import json
    from datetime import datetime, timezone

    print("[temporal] Discovering temporal relationships...")

    # Load nodes
    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()
    g_cursor.execute("SELECT id, properties FROM nodes")
    nodes = {row[0]: json.loads(row[1]) for row in g_cursor.fetchall()}
    g_conn.close()

    # Load embedding timestamps
    v_conn = sqlite3.connect(vectors_db_path)
    v_cursor = v_conn.cursor()

    node_timestamps = {}
    for node_id, properties in nodes.items():
        embedding_ids = properties.get("embedding_ids", [])
        if not embedding_ids:
            continue

        placeholders = ",".join("?" * len(embedding_ids))
        v_cursor.execute(
            f"""
            SELECT created_at
            FROM embeddings
            WHERE id IN ({placeholders})
        """,
            embedding_ids,
        )

        timestamps = []
        for row in v_cursor.fetchall():
            timestamp_str = row[0]
            # Handle both timezone-aware and naive timestamps
            if "Z" in timestamp_str or "+" in timestamp_str:
                # Timezone-aware
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                # Timezone-naive, make it UTC-aware
                dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            timestamps.append(dt)

        if timestamps:
            node_timestamps[node_id] = {
                "earliest": min(timestamps),
                "latest": max(timestamps),
                "count": len(timestamps),
            }

    v_conn.close()

    # Find temporal relationships
    relationships = []
    node_ids = list(node_timestamps.keys())

    for i, source_id in enumerate(node_ids):
        for target_id in node_ids[i + 1 :]:
            source_times = node_timestamps[source_id]
            target_times = node_timestamps[target_id]

            # Check if source precedes target
            if source_times["latest"] < target_times["earliest"]:
                time_gap = (
                    target_times["earliest"] - source_times["latest"]
                ).total_seconds() / 3600
                if time_gap <= time_window_hours:
                    # Weight: inverse of time gap (normalized)
                    weight = 1.0 - (time_gap / time_window_hours)
                    relationships.append((source_id, target_id, weight))
                    print(
                        f"  [v] {source_id} -> {target_id} (gap: {time_gap:.1f}h, weight: {weight:.2f})"
                    )

            # Check if target precedes source
            elif target_times["latest"] < source_times["earliest"]:
                time_gap = (
                    source_times["earliest"] - target_times["latest"]
                ).total_seconds() / 3600
                if time_gap <= time_window_hours:
                    weight = 1.0 - (time_gap / time_window_hours)
                    relationships.append((target_id, source_id, weight))
                    print(
                        f"  [v] {target_id} -> {source_id} (gap: {time_gap:.1f}h, weight: {weight:.2f})"
                    )

    print(f"[v] Found {len(relationships)} temporal relationships")
    return relationships


def discover_semantic_relationships(
    graph_db_path: str, vectors_db_path: str, similarity_threshold: float = 0.65
) -> List[Tuple[str, str, float]]:
    """
    Discover semantic relationships between nodes based on embedding similarity.

    Computes centroid (average embedding) for each node and finds pairs
    with high cosine similarity, indicating semantic relatedness.

    Args:
        graph_db_path: Path to graph.db
        vectors_db_path: Path to vectors.db
        similarity_threshold: Minimum similarity to create edge (0-1)

    Returns:
        List of (source_node_id, target_node_id, similarity) tuples
        Similarity is cosine similarity (0-1, higher = more similar)
    """
    import json

    from scipy.spatial.distance import cosine

    print("[semantic] Discovering semantic relationships...")

    # Load nodes
    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()
    g_cursor.execute("SELECT id, properties FROM nodes")
    nodes = {row[0]: json.loads(row[1]) for row in g_cursor.fetchall()}
    g_conn.close()

    # Load embeddings and compute centroids
    v_conn = sqlite3.connect(vectors_db_path)
    v_cursor = v_conn.cursor()

    node_centroids = {}
    for node_id, properties in nodes.items():
        embedding_ids = properties.get("embedding_ids", [])
        if not embedding_ids:
            continue

        placeholders = ",".join("?" * len(embedding_ids))
        v_cursor.execute(
            f"""
            SELECT embedding
            FROM embeddings
            WHERE id IN ({placeholders})
            AND embedding IS NOT NULL
        """,
            embedding_ids,
        )

        embeddings = [
            np.frombuffer(row[0], dtype=np.float32) for row in v_cursor.fetchall()
        ]

        if embeddings:
            # Compute centroid (average of all embeddings)
            centroid = np.mean(embeddings, axis=0)
            node_centroids[node_id] = centroid

    v_conn.close()

    # Find semantic relationships
    relationships = []
    node_ids = list(node_centroids.keys())

    for i, source_id in enumerate(node_ids):
        for target_id in node_ids[i + 1 :]:
            source_centroid = node_centroids[source_id]
            target_centroid = node_centroids[target_id]

            # Compute cosine similarity
            similarity = 1.0 - cosine(source_centroid, target_centroid)

            if similarity >= similarity_threshold:
                relationships.append((source_id, target_id, similarity))
                print(f"  [v] {source_id} <-> {target_id} (similarity: {similarity:.3f})")

    print(f"[v] Found {len(relationships)} semantic relationships")
    return relationships


def create_graph_edges(
    relationships: List[Tuple[str, str, float]],
    relationship_type: str,
    graph_db_path: str,
) -> int:
    """
    Create edges in graph.db from discovered relationships.

    Args:
        relationships: List of (source_id, target_id, weight) tuples
        relationship_type: Type of relationship (e.g., 'temporal', 'semantic')
        graph_db_path: Path to graph.db

    Returns:
        Number of edges created
    """
    if not relationships:
        print(f"[!]  No {relationship_type} relationships to create")
        return 0

    print(f"[db] Creating {len(relationships)} {relationship_type} edges...")

    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()

    created = 0
    skipped = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for source_id, target_id, weight in relationships:
        # Use INSERT OR IGNORE to skip existing edges (same source, target, type)
        # This is cleaner than INSERT OR REPLACE which would delete and re-insert
        g_cursor.execute(
            """
            INSERT OR IGNORE INTO relationships (
                source_id, target_id, relationship_type, weight, metadata,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (source_id, target_id, relationship_type, weight, json.dumps({}),
             timestamp, timestamp),
        )
        if g_cursor.rowcount > 0:
            created += 1
        else:
            skipped += 1
            print(f"  [#] Exists: {source_id} -> {target_id}")

    g_conn.commit()
    g_conn.close()

    print(f"[v] Created {created} {relationship_type} edges" +
          (f" (skipped {skipped} existing)" if skipped > 0 else ""))
    return created


def discover_causal_relationships(
    graph_db_path: str,
    metadata_db_path: str,
    vectors_db_path: str,
    temporal_weight: float = 0.5,
    semantic_weight: float = 0.5,
    min_combined_score: float = 0.70,
) -> List[Tuple[str, str, float]]:
    """
    Discover causal relationships by combining temporal and semantic signals.

    Causal inference requires both:
    1. Temporal precedence: A happens before B (necessary condition)
    2. Semantic relevance: A and B are conceptually related (relevance condition)

    Combined score = (temporal_weight × temporal_score) + (semantic_weight × semantic_score)

    Args:
        graph_db_path: Path to graph.db
        metadata_db_path: Path to metadata.db
        vectors_db_path: Path to vectors.db
        temporal_weight: Weight for temporal component (default 0.5)
        semantic_weight: Weight for semantic component (default 0.5)
        min_combined_score: Minimum score to create causal edge (default 0.70)

    Returns:
        List of (source_node_id, target_node_id, causal_strength) tuples
        causal_strength is combined score (0-1, higher = stronger causal evidence)
    """
    import json
    from datetime import datetime, timezone

    from scipy.spatial.distance import cosine

    print("[causal] Discovering causal relationships...")
    print(f"   Weights: temporal={temporal_weight}, semantic={semantic_weight}")
    print(f"   Threshold: {min_combined_score}")

    # Load nodes
    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()
    g_cursor.execute("SELECT id, properties FROM nodes")
    nodes = {row[0]: json.loads(row[1]) for row in g_cursor.fetchall()}
    g_conn.close()

    # Load embedding vectors and timestamps
    v_conn = sqlite3.connect(vectors_db_path)
    v_cursor = v_conn.cursor()

    # Compute node data: timestamps + centroids
    node_data = {}
    for node_id, properties in nodes.items():
        embedding_ids = properties.get("embedding_ids", [])
        if not embedding_ids:
            continue

        placeholders = ",".join("?" * len(embedding_ids))
        v_cursor.execute(
            f"""
            SELECT created_at, embedding
            FROM embeddings
            WHERE id IN ({placeholders})
            AND embedding IS NOT NULL
        """,
            embedding_ids,
        )

        timestamps = []
        embeddings = []
        for row in v_cursor.fetchall():
            # Parse timestamp
            timestamp_str = row[0]
            if "Z" in timestamp_str or "+" in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
            timestamps.append(dt)

            # Parse embedding
            embeddings.append(np.frombuffer(row[1], dtype=np.float32))

        if timestamps and embeddings:
            node_data[node_id] = {
                "earliest": min(timestamps),
                "latest": max(timestamps),
                "centroid": np.mean(embeddings, axis=0),
            }

    v_conn.close()

    # Discover causal relationships
    relationships = []
    node_ids = list(node_data.keys())

    for i, source_id in enumerate(node_ids):
        for target_id in node_ids[i + 1 :]:
            source_data = node_data[source_id]
            target_data = node_data[target_id]

            # Check temporal precedence (bidirectional)
            temporal_score = 0.0
            direction = None

            if source_data["latest"] < target_data["earliest"]:
                # Source precedes target
                time_gap_hours = (
                    target_data["earliest"] - source_data["latest"]
                ).total_seconds() / 3600
                # Normalize to [0, 1]: closer in time = higher score
                # Use 1 week (168 hours) as reference window
                temporal_score = max(0.0, 1.0 - (time_gap_hours / 168.0))
                direction = (source_id, target_id)

            elif target_data["latest"] < source_data["earliest"]:
                # Target precedes source
                time_gap_hours = (
                    source_data["earliest"] - target_data["latest"]
                ).total_seconds() / 3600
                temporal_score = max(0.0, 1.0 - (time_gap_hours / 168.0))
                direction = (target_id, source_id)

            # Skip if no temporal precedence
            if direction is None or temporal_score == 0.0:
                continue

            # Compute semantic similarity
            semantic_score = 1.0 - cosine(
                source_data["centroid"], target_data["centroid"]
            )

            # Compute combined causal score
            causal_score = (temporal_weight * temporal_score) + (
                semantic_weight * semantic_score
            )

            if causal_score >= min_combined_score:
                relationships.append((direction[0], direction[1], causal_score))
                print(f"  [v] {direction[0]} ⇒ {direction[1]}")
                print(
                    f"     temporal={temporal_score:.3f}, semantic={semantic_score:.3f}, causal={causal_score:.3f}"
                )

    print(f"[v] Found {len(relationships)} causal relationships")
    return relationships


def build_knowledge_graph(
    vectors_db_path: str,
    metadata_db_path: str,
    graph_db_path: str,
    rebuild: bool = False,
    normalize_weights: bool = True,
) -> Dict[str, any]:
    """
    Unified function to build complete knowledge graph from embeddings.

    Pipeline:
    1. Cluster embeddings -> concept nodes
    2. Create graph nodes
    3. Discover temporal relationships
    4. Discover semantic relationships
    5. Discover causal relationships
    6. Create all edges
    7. Normalize weights (optional)

    Args:
        vectors_db_path: Path to vectors.db
        metadata_db_path: Path to metadata.db
        graph_db_path: Path to graph.db
        rebuild: If True, clear existing graph before building
        normalize_weights: If True, normalize weights to [0, 1] per type

    Returns:
        Dict with build statistics
    """
    print("[graph] Building Knowledge Graph...")
    print("=" * 80)

    stats = {
        "clusters": 0,
        "nodes": 0,
        "edges": {"temporal": 0, "semantic": 0, "causal": 0, "total": 0},
        "weights": {"normalized": normalize_weights},
    }

    # Step 1: Cluster embeddings
    print("\n[#] Step 1: Clustering embeddings...")
    clusters = cluster_embeddings(
        vectors_db_path=vectors_db_path, metadata_db_path=metadata_db_path, limit=None
    )
    stats["clusters"] = len(clusters)
    print(f"  [v] Created {stats['clusters']} clusters")

    # Step 2: Create nodes (clear if rebuild)
    print("\n[^]  Step 2: Creating graph nodes...")
    if rebuild:
        deleted = clear_graph_nodes(graph_db_path)
        print(f"  [v] Cleared {deleted} existing nodes")

    node_mapping = create_graph_nodes(
        cluster_data=clusters,
        vectors_db_path=vectors_db_path,
        metadata_db_path=metadata_db_path,
        graph_db_path=graph_db_path,
    )
    stats["nodes"] = len(node_mapping)

    # Clear existing edges if rebuild
    if rebuild:
        g_conn = sqlite3.connect(graph_db_path)
        g_cursor = g_conn.cursor()
        g_cursor.execute("DELETE FROM relationships")
        g_conn.commit()
        g_conn.close()
        print(f"  [v] Cleared existing edges")

    # Step 3: Discover temporal relationships
    print("\n[Step 3] Discovering temporal relationships...")
    temporal_rels = discover_temporal_relationships(
        graph_db_path=graph_db_path,
        metadata_db_path=metadata_db_path,
        vectors_db_path=vectors_db_path,
        time_window_hours=168.0,
    )
    stats["edges"]["temporal"] = create_graph_edges(
        temporal_rels, "temporal", graph_db_path
    )

    # Step 4: Discover semantic relationships
    print("\n[Step 4] Discovering semantic relationships...")
    semantic_rels = discover_semantic_relationships(
        graph_db_path=graph_db_path,
        vectors_db_path=vectors_db_path,
        similarity_threshold=0.65,
    )
    stats["edges"]["semantic"] = create_graph_edges(
        semantic_rels, "semantic", graph_db_path
    )

    # Step 5: Discover causal relationships
    print("\n[Step 5] Discovering causal relationships...")
    causal_rels = discover_causal_relationships(
        graph_db_path=graph_db_path,
        metadata_db_path=metadata_db_path,
        vectors_db_path=vectors_db_path,
        temporal_weight=0.5,
        semantic_weight=0.5,
        min_combined_score=0.70,
    )
    stats["edges"]["causal"] = create_graph_edges(causal_rels, "causal", graph_db_path)

    # Step 6: Normalize weights (optional)
    if normalize_weights:
        print("\n[=]  Step 6: Normalizing edge weights...")
        normalize_edge_weights(graph_db_path)
        print("  [v] Weights normalized per relationship type")

    # Compute total edges
    stats["edges"]["total"] = sum(
        [
            stats["edges"]["temporal"],
            stats["edges"]["semantic"],
            stats["edges"]["causal"],
        ]
    )

    print("\n" + "=" * 80)
    print("[v] Knowledge Graph Build Complete")
    print("=" * 80)
    print(f"   Nodes: {stats['nodes']}")
    print(f"   Edges: {stats['edges']['total']} total")
    print(f"     - Temporal: {stats['edges']['temporal']}")
    print(f"     - Semantic: {stats['edges']['semantic']}")
    print(f"     - Causal: {stats['edges']['causal']}")
    print("=" * 80)

    return stats


def normalize_edge_weights(graph_db_path: str) -> Dict[str, Dict[str, float]]:
    """
    Normalize edge weights to [0, 1] per relationship type.

    Each relationship type may have different weight distributions.
    Normalization ensures comparable weights across types.

    Args:
        graph_db_path: Path to graph.db

    Returns:
        Dict of {relationship_type: {min, max, mean}} before normalization
    """
    import struct

    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()

    # Get all relationship types
    g_cursor.execute("SELECT DISTINCT relationship_type FROM relationships")
    rel_types = [row[0] for row in g_cursor.fetchall()]

    stats = {}

    for rel_type in rel_types:
        # Get all weights for this type
        g_cursor.execute(
            """
            SELECT id, weight FROM relationships
            WHERE relationship_type = ?
        """,
            (rel_type,),
        )

        weights = []
        edge_ids = []
        for row in g_cursor.fetchall():
            edge_id, weight = row
            # Handle SQLite REAL/bytes issue
            if isinstance(weight, bytes):
                weight = struct.unpack("f", weight)[0]
            weights.append(weight)
            edge_ids.append(edge_id)

        if not weights:
            continue

        # Compute statistics before normalization
        min_weight = min(weights)
        max_weight = max(weights)
        mean_weight = sum(weights) / len(weights)

        stats[rel_type] = {
            "min": min_weight,
            "max": max_weight,
            "mean": mean_weight,
            "count": len(weights),
        }

        # Normalize to [0, 1]
        weight_range = max_weight - min_weight
        if weight_range > 0:
            for i, edge_id in enumerate(edge_ids):
                normalized = (weights[i] - min_weight) / weight_range
                g_cursor.execute(
                    """
                    UPDATE relationships
                    SET weight = ?
                    WHERE id = ?
                """,
                    (normalized, edge_id),
                )

    g_conn.commit()
    g_conn.close()

    return stats


def update_edge_metadata(
    graph_db_path: str,
    source_id: str,
    target_id: str,
    relationship_type: str,
    metadata_updates: Dict[str, any],
) -> bool:
    """
    Update metadata for a specific edge (non-destructive).

    Args:
        graph_db_path: Path to graph.db
        source_id: Source node ID
        target_id: Target node ID
        relationship_type: Type of relationship
        metadata_updates: Dict of key-value pairs to add/update

    Returns:
        True if edge was found and updated, False otherwise
    """
    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()

    # Check if edge exists
    g_cursor.execute(
        """
        SELECT id, metadata FROM relationships
        WHERE source_id = ? AND target_id = ? AND relationship_type = ?
    """,
        (source_id, target_id, relationship_type),
    )

    row = g_cursor.fetchone()
    if not row:
        g_conn.close()
        return False

    edge_id, metadata_json = row
    metadata = json.loads(metadata_json) if metadata_json else {}

    # Update metadata
    metadata.update(metadata_updates)

    # Save back
    g_cursor.execute(
        """
        UPDATE relationships
        SET metadata = ?
        WHERE id = ?
    """,
        (json.dumps(metadata), edge_id),
    )

    g_conn.commit()
    g_conn.close()

    return True


def get_edge_weight(
    graph_db_path: str, source_id: str, target_id: str, relationship_type: str
) -> Optional[float]:
    """
    Get weight for a specific edge.

    Args:
        graph_db_path: Path to graph.db
        source_id: Source node ID
        target_id: Target node ID
        relationship_type: Type of relationship

    Returns:
        Weight (float) or None if edge doesn't exist
    """
    import struct

    g_conn = sqlite3.connect(graph_db_path)
    g_cursor = g_conn.cursor()

    g_cursor.execute(
        """
        SELECT weight FROM relationships
        WHERE source_id = ? AND target_id = ? AND relationship_type = ?
    """,
        (source_id, target_id, relationship_type),
    )

    row = g_cursor.fetchone()
    g_conn.close()

    if not row:
        return None

    weight = row[0]
    # Handle SQLite REAL/bytes issue
    if isinstance(weight, bytes):
        weight = struct.unpack("f", weight)[0]

    return weight


# ============================================================================
# END GRAPH BUILDING FUNCTIONS
# ============================================================================


# ============================================================================
# HELPER FUNCTIONS FOR CLI
# ============================================================================

def _detect_cross_refs(cam: CAM, file_path: str, emb_id: str) -> int:
    """
    Detect and create cross-reference relationships for markdown files.

    Args:
        cam: CAM instance
        file_path: Path to the markdown file
        emb_id: Embedding ID of the file

    Returns:
        Number of cross-references created
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return 0

    # Find markdown links: [text](path.md) or [text](../path.md)
    link_pattern = r'\[([^\]]+)\]\(([^)]+\.md)\)'
    links = re.findall(link_pattern, content)

    if not links:
        return 0

    cursor = cam.vectors_conn.cursor()
    refs_created = 0

    for link_text, link_path in links:
        # Skip external links
        if link_path.startswith("http"):
            continue

        # Resolve relative path
        base_dir = os.path.dirname(file_path)
        resolved_path = os.path.normpath(os.path.join(base_dir, link_path))
        link_basename = os.path.basename(resolved_path)

        # Search for target document in CAM
        cursor.execute(
            "SELECT id FROM embeddings WHERE source_file LIKE ? AND source_type = 'docs'",
            (f"%{link_basename}",)
        )
        row = cursor.fetchone()

        if row:
            target_id = row[0]
            # Create "references" relationship
            try:
                cam.add_relationship(
                    source_id=emb_id,
                    target_id=target_id,
                    relationship_type="references",
                    weight=0.8,
                    metadata={
                        "link_text": link_text,
                        "link_path": link_path,
                        "detected_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                refs_created += 1
                print(f"  [v] {emb_id[:8]} --references--> {target_id[:8]} ({link_basename})")
            except Exception:
                pass  # Relationship may already exist

    if refs_created > 0:
        print(f"[v] Created {refs_created} cross-reference relationships for {os.path.basename(file_path)}")

    return refs_created


def main():
    """CLI interface for CAM"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: cam_core.py <command> [args...]")
        print("")
        print("Core Commands:")
        print("  version                  - Show CAM version")
        print("  stats                    - Show CAM statistics")
        print("  query <text> [--type T]  - Semantic search (optional type filter)")
        print("  get <id>                 - Retrieve full content by embedding ID")
        print("  annotate <content>       - Add manual annotation to CAM")
        print("  graph build [--mode MODE] - Build knowledge graph (v1.1.0)")
        print("")
        print("Session Commands (v1.5.2):")
        print("  sessions [--limit N]     - List recent sessions")
        print("  last-session             - Get most recent session summary")
        print("  session <id>             - Get specific session by ID")
        print("  store-session <id> <json> - Store session summary (hook use)")
        print("")
        print("Ingestion Commands (v1.5.0):")
        print("  ingest <path> <type>     - Ingest file OR directory into CAM")
        print("                             Supports: code, docs, config")
        print("  ingest <path> [--force]  - Auto-detect type, --force re-ingests unchanged")
        print("  scan <directory>         - Scan directory, show new/modified/unchanged files")
        print("  check-file <path>        - Check single file status (new/modified/unchanged)")
        print("  file-stats               - Show file index statistics")
        print("")
        print("Evaluation Commands (v1.2.0):")
        print(
            "  eval embeddings          - Evaluate embedding quality (STS correlation)"
        )
        print(
            "  eval retrieval           - Evaluate retrieval accuracy (precision/recall/nDCG)"
        )
        print("  eval graph               - Evaluate knowledge graph quality")
        print("  eval all                 - Run all intrinsic evaluations")
        print(
            "  eval extrinsic           - Run extrinsic evaluation (operational metrics)"
        )
        print("  benchmark dmr            - Run Deep Memory Retrieval benchmark")
        print("  benchmark locomo         - Run LoCoMo-style benchmark")
        print("  benchmark all            - Run all benchmarks")
        sys.exit(1)

    command = sys.argv[1]

    if command == "version":
        print(CAM_VERSION)
        sys.exit(0)

    with CAM() as cam:
        if command == "stats":
            stats = cam.stats()
            print(json.dumps(stats, indent=2))

        elif command == "query":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py query <text> [top_k] [--type TYPE]")
                print("  --type TYPE  Filter by metadata type (e.g., session_summary, ephemeral_note)")
                sys.exit(1)

            query_text = sys.argv[2]
            top_k = 5
            metadata_type = None

            # Parse remaining arguments
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--type" and i + 1 < len(sys.argv):
                    metadata_type = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i].isdigit():
                    top_k = int(sys.argv[i])
                    i += 1
                else:
                    i += 1

            # Use type-filtered query if type specified
            if metadata_type:
                results = cam.query_by_metadata_type(query_text, metadata_type, top_k=top_k)
            else:
                results = cam.query(query_text, top_k=top_k)

            for i, (emb_id, content, score) in enumerate(results, 1):
                print(f"\n{i}. [Score: {score:.4f}] {emb_id}")
                print(f"   {content[:200]}...")

        elif command == "get":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py get <embedding_id>")
                print("")
                print("Retrieves the full content of an embedding by its ID.")
                print("Use 'query' first to find embedding IDs, then 'get' to see full content.")
                sys.exit(1)

            embedding_id = sys.argv[2]
            result = cam.get_embedding(embedding_id)

            if result:
                print(json.dumps(result, indent=2))
            else:
                print(f"[!] No embedding found with ID: {embedding_id}")
                sys.exit(1)

        elif command == "ingest":
            # =========================================================
            # PHASE 5: Enhanced ingest with directory support (v1.5.0)
            # =========================================================
            if len(sys.argv) < 3:
                print("Usage: cam_core.py ingest <path> [type] [--force] [--dry-run]")
                print("")
                print("Arguments:")
                print("  path      - File or directory to ingest")
                print("  type      - Source type: code, docs, config (auto-detected if omitted)")
                print("  --force   - Re-ingest even if file is unchanged")
                print("  --dry-run - Show what would be ingested without doing it")
                print("")
                print("Examples:")
                print("  cam_core.py ingest src/main.py code")
                print("  cam_core.py ingest .ai/ docs")
                print("  cam_core.py ingest . --dry-run")
                print("  cam_core.py ingest src/ code --force")
                sys.exit(1)

            target_path = sys.argv[2]
            source_type = None
            force = "--force" in sys.argv
            dry_run = "--dry-run" in sys.argv

            # Parse source type if provided (not a flag)
            for arg in sys.argv[3:]:
                if not arg.startswith("--") and arg in ['code', 'docs', 'config']:
                    source_type = arg
                    break

            # Check if path is directory or file
            if os.path.isdir(target_path):
                # Directory ingestion
                print(f"[~] Scanning directory: {target_path}")
                stats = cam.ingest_directory(target_path, source_type, force=force, dry_run=dry_run)

                if dry_run:
                    print(f"\n[DRY RUN] Would ingest {len(stats.get('would_ingest', []))} files:")
                    for f in stats.get('would_ingest', [])[:20]:
                        print(f"  - {f}")
                    if len(stats.get('would_ingest', [])) > 20:
                        print(f"  ... and {len(stats['would_ingest']) - 20} more")
                else:
                    print(f"\n[v] Ingestion complete:")
                    print(f"    Scanned: {stats['scanned']}")
                    print(f"    New: {stats['new']}")
                    print(f"    Modified: {stats['modified']}")
                    print(f"    Unchanged: {stats['unchanged']}")
                    print(f"    Ignored: {stats['ignored']}")
                    print(f"    Ingested: {stats['ingested']}")
                    if stats['failed'] > 0:
                        print(f"    Failed: {stats['failed']}")

                    # Run cross-reference detection for docs
                    if source_type == 'docs' or source_type is None:
                        print("\n[^] Running cross-reference detection...")
                        for file_info in stats.get('files', []):
                            if file_info['status'] == 'success' and file_info['path'].endswith('.md'):
                                _detect_cross_refs(cam, file_info['path'], file_info['id'])

            elif os.path.isfile(target_path):
                # Single file ingestion
                emb_id = cam.ingest_file(target_path, source_type, force=force)

                if emb_id:
                    print(f"[v] Ingested {target_path} -> {emb_id}")

                    # Cross-reference detection for markdown
                    if target_path.endswith('.md'):
                        _detect_cross_refs(cam, target_path, emb_id)
                else:
                    status, _ = cam.check_file_status(target_path)
                    if status == 'unchanged':
                        print(f"[#] File unchanged, skipping: {target_path}")
                        print("    Use --force to re-ingest")
                    else:
                        print(f"[!] Failed to ingest: {target_path}")
                        sys.exit(1)
            else:
                print(f"[!] Path not found: {target_path}")
                sys.exit(1)

        # =========================================================
        # PHASE 5: New scan command (v1.5.0)
        # =========================================================
        elif command == "scan":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py scan <directory> [type]")
                print("")
                print("Scan directory and report file status (new/modified/unchanged)")
                print("")
                print("Arguments:")
                print("  directory - Directory to scan")
                print("  type      - Optional filter: code, docs, config")
                sys.exit(1)

            directory = sys.argv[2]
            source_type = sys.argv[3] if len(sys.argv) > 3 else None

            if not os.path.isdir(directory):
                print(f"[!] Not a directory: {directory}")
                sys.exit(1)

            print(f"[~] Scanning: {directory}")
            results = cam.scan_directory(directory, source_type)

            print(f"\n--- Scan Results ---")
            print(f"New files:       {len(results['new'])}")
            print(f"Modified files:  {len(results['modified'])}")
            print(f"Unchanged files: {len(results['unchanged'])}")
            print(f"Ignored files:   {len(results['ignored'])}")
            print(f"Skipped (size):  {len(results['skipped'])}")

            if results['new']:
                print(f"\n[NEW] ({len(results['new'])} files)")
                for f in results['new'][:10]:
                    print(f"  + {f}")
                if len(results['new']) > 10:
                    print(f"  ... and {len(results['new']) - 10} more")

            if results['modified']:
                print(f"\n[MODIFIED] ({len(results['modified'])} files)")
                for f in results['modified'][:10]:
                    print(f"  ~ {f}")
                if len(results['modified']) > 10:
                    print(f"  ... and {len(results['modified']) - 10} more")

        # =========================================================
        # PHASE 5: New check-file command (v1.5.0)
        # =========================================================
        elif command == "check-file":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py check-file <path>")
                sys.exit(1)

            file_path = sys.argv[2]
            status, stored_hash = cam.check_file_status(file_path)

            print(f"File: {file_path}")
            print(f"Status: {status}")

            if status == 'unchanged':
                entry = cam.get_file_index_entry(file_path)
                if entry:
                    print(f"Last ingested: {entry.last_ingested_at}")
                    print(f"Embedding ID: {entry.embedding_id}")
                    print(f"Source type: {entry.source_type}")

        # =========================================================
        # PHASE 5: New file-stats command (v1.5.0)
        # =========================================================
        elif command == "file-stats":
            stats = cam.file_index_stats()
            print(json.dumps(stats, indent=2))

        elif command == "annotate":
            if len(sys.argv) < 3:
                print(
                    "Usage: cam_core.py annotate '<content>' [--metadata '<json>'] [--tags '<tag1,tag2>']"
                )
                sys.exit(1)

            content = sys.argv[2]
            metadata = {}
            tags = []

            # Parse optional arguments
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--metadata" and i + 1 < len(sys.argv):
                    try:
                        metadata = json.loads(sys.argv[i + 1])
                    except json.JSONDecodeError as e:
                        print(f"[!] Invalid JSON for --metadata: {e}")
                        print(f"    Received: {sys.argv[i + 1][:100]}...")
                        print(f"    Tip: Ensure JSON is properly quoted for shell")
                        print(f"    Example: --metadata '{{\"key\": \"value\"}}'")
                        sys.exit(1)
                    i += 2
                elif sys.argv[i] == "--tags" and i + 1 < len(sys.argv):
                    tags = [tag.strip() for tag in sys.argv[i + 1].split(",")]
                    i += 2
                else:
                    i += 1

            # Store embedding
            emb_id = cam.store_embedding(
                content=content, source_type="operation", generate_embedding=True
            )

            # Add annotation if provided
            if metadata or tags:
                cam.annotate(emb_id, metadata=metadata, tags=tags)

            print(f"[v] Annotation stored: {emb_id}")

        # =====================================================================
        # EVALUATION COMMANDS (v1.2.0)
        # =====================================================================
        elif command == "eval":
            if len(sys.argv) < 3:
                print(
                    "Usage: cam_core.py eval <embeddings|retrieval|graph|all|extrinsic>"
                )
                sys.exit(1)

            eval_type = sys.argv[2]
            evaluator = CAMEvaluator(cam)

            if eval_type == "embeddings":
                result = evaluator.eval_embeddings()
                print(json.dumps(result.to_dict(), indent=2))

            elif eval_type == "retrieval":
                result = evaluator.eval_retrieval()
                print(json.dumps(result.to_dict(), indent=2))

            elif eval_type == "graph":
                result = evaluator.eval_graph()
                print(json.dumps(result.to_dict(), indent=2))

            elif eval_type == "all":
                results = evaluator.run_all_intrinsic()
                print("\n[#] Detailed Results:")
                for name, result in results.items():
                    print(f"\n--- {name.upper()} ---")
                    print(json.dumps(result.to_dict(), indent=2))

            elif eval_type == "extrinsic":
                result = evaluator.eval_extrinsic()
                print(json.dumps(result.to_dict(), indent=2))

            else:
                print(f"Unknown eval type: {eval_type}")
                print("Available: embeddings, retrieval, graph, all, extrinsic")
                sys.exit(1)

        elif command == "benchmark":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py benchmark <dmr|locomo|all>")
                sys.exit(1)

            benchmark_type = sys.argv[2]
            evaluator = CAMEvaluator(cam)

            if benchmark_type == "dmr":
                result = evaluator.benchmark_dmr()
                print(json.dumps(result.to_dict(), indent=2))

            elif benchmark_type == "locomo":
                result = evaluator.benchmark_locomo()
                print(json.dumps(result.to_dict(), indent=2))

            elif benchmark_type == "all":
                results = evaluator.run_all_benchmarks()
                print("\n[#] Benchmark Results:")
                for name, result in results.items():
                    print(f"\n--- {name.upper()} ---")
                    print(json.dumps(result.to_dict(), indent=2))

            else:
                print(f"Unknown benchmark: {benchmark_type}")
                print("Available: dmr, locomo, all")
                sys.exit(1)

        # =====================================================================
        # GRAPH COMMANDS (v1.1.0)
        # =====================================================================
        elif command == "graph":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py graph <build|stats>")
                print("  build [--rebuild]  - Build knowledge graph from embeddings")
                print("  stats              - Show graph statistics")
                sys.exit(1)

            graph_cmd = sys.argv[2]

            if graph_cmd == "build":
                rebuild = "--rebuild" in sys.argv
                if rebuild:
                    print("[~] Rebuilding graph from scratch...")
                stats = build_knowledge_graph(
                    vectors_db_path=VECTORS_DB,
                    metadata_db_path=METADATA_DB,
                    graph_db_path=GRAPH_DB,
                    rebuild=rebuild,
                )
                print(json.dumps(stats, indent=2))

            elif graph_cmd == "stats":
                g_conn = sqlite3.connect(GRAPH_DB)
                g_cursor = g_conn.cursor()
                g_cursor.execute("SELECT COUNT(*) FROM nodes")
                node_count = g_cursor.fetchone()[0]
                g_cursor.execute("SELECT COUNT(*) FROM relationships")
                edge_count = g_cursor.fetchone()[0]
                g_cursor.execute(
                    "SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type"
                )
                edge_types = dict(g_cursor.fetchall())
                g_conn.close()

                print(
                    json.dumps(
                        {
                            "nodes": node_count,
                            "edges": edge_count,
                            "edge_types": edge_types,
                        },
                        indent=2,
                    )
                )

            else:
                print(f"Unknown graph command: {graph_cmd}")
                sys.exit(1)

        # =====================================================================
        # RELATIONSHIP COMMANDS (v1.4.0)
        # =====================================================================
        elif command == "relate":
            if len(sys.argv) < 5:
                print("Usage: cam_core.py relate <source_id> <target_id> <type> [weight]")
                print("")
                print("Create a relationship between two embeddings.")
                print("")
                print("Arguments:")
                print("  source_id  - Source embedding ID (from query or ingest)")
                print("  target_id  - Target embedding ID")
                print("  type       - Relationship type (e.g., modifies, references, affects)")
                print("  weight     - Optional weight 0.0-1.0 (default: 1.0)")
                print("")
                print("Examples:")
                print("  cam_core.py relate abc123 def456 modifies")
                print("  cam_core.py relate abc123 def456 references 0.8")
                sys.exit(1)

            source_id = sys.argv[2]
            target_id = sys.argv[3]
            rel_type = sys.argv[4]
            weight = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

            # Validate IDs exist
            source_exists = cam.get_embedding(source_id) is not None
            target_exists = cam.get_embedding(target_id) is not None

            if not source_exists:
                print(f"[!] Source embedding not found: {source_id}")
                sys.exit(1)
            if not target_exists:
                print(f"[!] Target embedding not found: {target_id}")
                sys.exit(1)

            # Create relationship
            cam.add_relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                weight=weight,
                metadata={"created_by": "cli", "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            print(f"[v] Created relationship: {source_id} --{rel_type}--> {target_id} (weight={weight})")

        elif command == "find-doc":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py find-doc <file_path>")
                print("")
                print("Find embedding ID for a document by its source file path.")
                print("Returns the embedding ID if found, or exits with error if not found.")
                print("")
                print("Example:")
                print("  cam_core.py find-doc .ai/patterns/api-and-routing.md")
                sys.exit(1)

            file_path = sys.argv[2]

            # Search for embedding by source_file
            cursor = cam.vectors_conn.cursor()
            cursor.execute(
                "SELECT id FROM embeddings WHERE source_file LIKE ? AND source_type = 'docs'",
                (f"%{file_path}",)
            )
            row = cursor.fetchone()

            if row:
                print(row[0])
            else:
                # Try partial match on basename
                basename = os.path.basename(file_path)
                cursor.execute(
                    "SELECT id FROM embeddings WHERE source_file LIKE ? AND source_type = 'docs'",
                    (f"%{basename}",)
                )
                row = cursor.fetchone()
                if row:
                    print(row[0])
                else:
                    sys.exit(1)  # Not found - exit with error code

        # =====================================================================
        # REBUILD COMMANDS (v1.2.0)
        # =====================================================================
        elif command == "rebuild":
            if len(sys.argv) < 3:
                print("Usage: cam_core.py rebuild <graph|embeddings|all>")
                print("  graph       - Rebuild knowledge graph only")
                print("  embeddings  - Re-generate all embeddings (SLOW)")
                print("  all         - Full rebuild: embeddings + graph")
                sys.exit(1)

            rebuild_type = sys.argv[2]

            if rebuild_type == "graph":
                stats = build_knowledge_graph(
                    vectors_db_path=VECTORS_DB,
                    metadata_db_path=METADATA_DB,
                    graph_db_path=GRAPH_DB,
                    rebuild=True,
                )
                print(json.dumps(stats, indent=2))

            elif rebuild_type == "embeddings":
                print("[~] Re-generating all embeddings...")
                cursor = cam.vectors_conn.cursor()
                cursor.execute("SELECT id, content FROM embeddings")
                rows = cursor.fetchall()

                updated = 0
                for emb_id, content in rows:
                    new_embedding = cam.embed(content)
                    if new_embedding is not None:
                        cursor.execute(
                            """
                            UPDATE embeddings
                            SET embedding = ?, updated_at = ?
                            WHERE id = ?
                        """,
                            (
                                new_embedding.tobytes(),
                                datetime.now(timezone.utc).isoformat(),
                                emb_id,
                            ),
                        )
                        updated += 1
                        print(f"  [v] {updated}/{len(rows)} embeddings updated")

                cam.vectors_conn.commit()
                print(f"[v] Re-embedded {updated} items")

            elif rebuild_type == "all":
                print("[~] Full rebuild: embeddings + graph...")

                # Re-embed
                cursor = cam.vectors_conn.cursor()
                cursor.execute("SELECT id, content FROM embeddings")
                rows = cursor.fetchall()
                for emb_id, content in rows:
                    new_embedding = cam.embed(content)
                    if new_embedding is not None:
                        cursor.execute(
                            """
                            UPDATE embeddings
                            SET embedding = ?, updated_at = ?
                            WHERE id = ?
                        """,
                            (
                                new_embedding.tobytes(),
                                datetime.now(timezone.utc).isoformat(),
                                emb_id,
                            ),
                        )
                cam.vectors_conn.commit()
                print(f"[v] Re-embedded {len(rows)} items")

                # Rebuild graph
                stats = build_knowledge_graph(
                    vectors_db_path=VECTORS_DB,
                    metadata_db_path=METADATA_DB,
                    graph_db_path=GRAPH_DB,
                    rebuild=True,
                )
                print(json.dumps(stats, indent=2))

            else:
                print(f"Unknown rebuild type: {rebuild_type}")
                sys.exit(1)

        # =====================================================================
        # PHASE 6: Session Memory Commands (v1.5.2)
        # =====================================================================

        elif command == "sessions":
            # List recent sessions
            limit = 10
            if len(sys.argv) > 2 and sys.argv[2] == "--limit":
                limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10

            sessions = cam.get_sessions(limit=limit)

            if not sessions:
                print("No sessions found in CAM database.")
                sys.exit(0)

            print(json.dumps({"sessions": sessions}, indent=2))

        elif command == "last-session":
            # Get most recent session summary
            session = cam.get_last_session()

            if not session:
                print("No session summaries found.")
                print("Tip: Session summaries are created when sessions end.")
                sys.exit(1)

            # Print human-readable format
            if session.get('content'):
                print(session['content'])
            else:
                # Generate summary from data
                print(f"## Session {session.get('session_id', 'unknown')[:8]} - {session.get('project', 'unknown')}")
                print(f"**Date**: {session.get('end_time', 'unknown')[:10] if session.get('end_time') else 'unknown'}")

                ops = session.get('operations', {})
                total = sum(ops.values()) if isinstance(ops, dict) else session.get('operation_count', 0)
                print(f"**Operations**: {total}")

                print("\n### Files Modified")
                for f in session.get('files_modified', [])[:15]:
                    print(f"- {f}")

                if isinstance(ops, dict) and ops:
                    print("\n### Operations Breakdown")
                    for op_type, count in ops.items():
                        if count > 0:
                            print(f"- {op_type}: {count}")

        elif command == "session":
            # Get specific session by ID
            if len(sys.argv) < 3:
                print("Usage: cam_core.py session <session_id>")
                print("  session_id - Full or partial (8-char) session ID")
                sys.exit(1)

            session_id = sys.argv[2]
            session = cam.get_session_summary(session_id)

            if not session:
                print(f"Session '{session_id}' not found.")
                print("Tip: Use 'cam_core.py sessions' to list available sessions.")
                sys.exit(1)

            # Print human-readable format
            if session.get('content'):
                print(session['content'])
            else:
                print(f"## Session {session.get('session_id', 'unknown')[:8]}")
                print(f"**Project**: {session.get('project', 'unknown')}")
                print(f"**Operations**: {session.get('operation_count', 0)}")

                print("\n### Files Modified")
                for f in session.get('files_modified', [])[:15]:
                    print(f"- {f}")

        elif command == "store-session":
            # Store a session summary (used by session-end.sh hook)
            if len(sys.argv) < 4:
                print("Usage: cam_core.py store-session <session_id> <json_data>")
                print("  json_data - JSON with: project, operations, files_modified, key_activities")
                sys.exit(1)

            session_id = sys.argv[2]
            try:
                data = json.loads(sys.argv[3])
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
                sys.exit(1)

            emb_id = cam.store_session_summary(
                session_id=session_id,
                project=data.get('project', 'unknown'),
                operations=data.get('operations', {}),
                files_modified=data.get('files_modified', []),
                key_activities=data.get('key_activities', []),
                start_time=data.get('start_time'),
                end_time=data.get('end_time')
            )

            print(f"[v] Session summary stored: {emb_id}")

        # =====================================================================
        # Ralph Wiggum Integration Commands (v1.6.0)
        # =====================================================================

        elif command == "ralph-loops":
            # List recent Ralph loops
            # Usage: ./cam.sh ralph-loops [--limit N] [--outcome success|max_iterations|cancelled]
            limit = 10
            outcome = None

            i = 2
            while i < len(sys.argv):
                if sys.argv[i] == "--limit" and i + 1 < len(sys.argv):
                    try:
                        limit = int(sys.argv[i + 1])
                    except ValueError:
                        print(f"Invalid limit: {sys.argv[i + 1]}")
                        sys.exit(1)
                    i += 2
                elif sys.argv[i] == "--outcome" and i + 1 < len(sys.argv):
                    outcome = sys.argv[i + 1]
                    if outcome not in ('success', 'max_iterations', 'cancelled'):
                        print(f"Invalid outcome: {outcome}")
                        print("Valid outcomes: success, max_iterations, cancelled")
                        sys.exit(1)
                    i += 2
                elif sys.argv[i] == "--successful":
                    outcome = "success"
                    i += 1
                else:
                    i += 1

            loops = cam.get_ralph_loops(limit=limit, outcome=outcome)

            if not loops:
                print("No Ralph loops found.")
                if outcome:
                    print(f"  (filtered by outcome: {outcome})")
            else:
                print(f"Found {len(loops)} Ralph loop(s):\n")
                for loop in loops:
                    meta = loop['metadata']
                    print(f"Loop ID: {meta.get('loop_id', 'unknown')}")
                    print(f"  Project: {meta.get('project', '?')}")
                    print(f"  Iterations: {meta.get('iterations', '?')}")
                    print(f"  Outcome: {meta.get('outcome', '?')}")
                    print(f"  Date: {loop['created_at'][:10] if loop.get('created_at') else '?'}")
                    # Show first 150 chars of content
                    content_preview = loop.get('content', '')[:150].replace('\n', ' ')
                    print(f"  Summary: {content_preview}...")
                    print()

        elif command == "ralph-history":
            # View iterations for a specific Ralph loop
            # Usage: ./cam.sh ralph-history <loop_id>
            if len(sys.argv) < 3:
                print("Usage: cam_core.py ralph-history <loop_id>")
                print("  loop_id - The unique Ralph loop identifier")
                print("")
                print("Tip: Use 'cam_core.py ralph-loops' to see available loop IDs")
                sys.exit(1)

            loop_id = sys.argv[2]
            iterations = cam.get_ralph_iterations(loop_id)

            if not iterations:
                print(f"No iterations found for loop: {loop_id}")
                print("")
                print("Tip: Use 'cam_core.py ralph-loops' to see available loop IDs")
            else:
                print(f"Ralph Loop {loop_id} - {len(iterations)} iteration(s):\n")
                for it in iterations:
                    meta = it['metadata']
                    print(f"Iteration {meta.get('iteration', '?')}:")
                    print(f"  Outcome: {meta.get('outcome', '?')}")
                    print(f"  Project: {meta.get('project', '?')}")
                    print(f"  Time: {it.get('created_at', '?')}")
                    print()

        elif command == "ralph-patterns":
            # Semantic search for similar Ralph loops
            # Usage: ./cam.sh ralph-patterns "task description"
            if len(sys.argv) < 3:
                print("Usage: cam_core.py ralph-patterns \"task description\"")
                print("  Searches for Ralph loops that worked on similar tasks")
                sys.exit(1)

            task = " ".join(sys.argv[2:])
            patterns = cam.query_ralph_patterns(task)

            if not patterns:
                print("No relevant Ralph patterns found.")
                print("")
                print("Tip: Ralph patterns are stored when loops complete.")
                print("     Run some Ralph loops first to build up patterns.")
            else:
                print(f"Found {len(patterns)} relevant pattern(s) for: {task}\n")
                for emb_id, content, score in patterns:
                    print(f"[Score: {score:.4f}] {emb_id[:12]}")
                    # Show first 300 chars, preserving newlines for readability
                    preview = content[:300]
                    if len(content) > 300:
                        preview += "..."
                    print(f"{preview}")
                    print()

        elif command == "store-ralph-iteration":
            # Store a Ralph iteration (used by stop-hook.sh)
            # Usage: ./cam.sh store-ralph-iteration <loop_id> <iteration> <outcome> <project> [changes]
            if len(sys.argv) < 6:
                print("Usage: cam_core.py store-ralph-iteration <loop_id> <iteration> <outcome> <project> [changes]")
                sys.exit(1)

            loop_id = sys.argv[2]
            try:
                iteration = int(sys.argv[3])
            except ValueError:
                print(f"Invalid iteration number: {sys.argv[3]}")
                sys.exit(1)
            outcome = sys.argv[4]
            project = sys.argv[5]
            changes = sys.argv[6] if len(sys.argv) > 6 else None

            emb_id = cam.store_ralph_iteration(
                loop_id=loop_id,
                iteration=iteration,
                outcome=outcome,
                project=project,
                changes=changes
            )

            print(f"[v] Ralph iteration stored: {emb_id}")

        elif command == "store-ralph-loop":
            # Store a Ralph loop summary (used by stop-hook.sh on completion)
            # Usage: ./cam.sh store-ralph-loop <json_data>
            if len(sys.argv) < 3:
                print("Usage: cam_core.py store-ralph-loop <json_data>")
                print("  json_data - JSON with: loop_id, iterations, outcome, project, prompt, started_at, files_modified")
                sys.exit(1)

            try:
                data = json.loads(sys.argv[2])
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
                sys.exit(1)

            required = ['loop_id', 'iterations', 'outcome', 'project', 'prompt', 'started_at']
            missing = [f for f in required if f not in data]
            if missing:
                print(f"Missing required fields: {', '.join(missing)}")
                sys.exit(1)

            emb_id = cam.store_ralph_loop_summary(
                loop_id=data['loop_id'],
                total_iterations=data['iterations'],
                outcome=data['outcome'],
                project=data['project'],
                prompt=data['prompt'],
                started_at=data['started_at'],
                files_modified=data.get('files_modified', [])
            )

            print(f"[v] Ralph loop summary stored: {emb_id}")

        elif command == "primer-status":
            # Show status of session primers (Phase 6)
            # Usage: ./cam.sh primer-status
            import time as time_module

            primer_dir = os.path.expanduser("~/.claude/.session-primers")

            if not os.path.exists(primer_dir):
                print("No primers directory found")
                print(f"Directory: {primer_dir}")
                sys.exit(0)

            primers = glob.glob(f"{primer_dir}/*.primer")

            if not primers:
                print("No active primers")
                sys.exit(0)

            print("Active Session Primers:")
            print("-" * 50)

            for primer_path in primers:
                try:
                    with open(primer_path) as f:
                        primer = json.load(f)

                    age_seconds = time_module.time() - os.path.getmtime(primer_path)
                    age_minutes = int(age_seconds / 60)
                    expiry_minutes = (4 * 60) - age_minutes  # 4 hour expiry

                    print(f"Project: {primer.get('project', 'unknown')}")
                    print(f"Session: {primer.get('session_id', 'unknown')[:8]}...")
                    print(f"Created: {primer.get('created_at', 'unknown')}")
                    print(f"Trigger: {primer.get('trigger', 'unknown')}")
                    print(f"Age: {age_minutes} minutes")
                    print(f"Expires in: {max(0, expiry_minutes)} minutes")

                    # Show summary info
                    summary = primer.get('summary', {})
                    ops = summary.get('operations', {})
                    print(f"Operations: Edits={ops.get('edits', 0)}, Writes={ops.get('writes', 0)}, Bash={ops.get('bash', 0)}")

                    files = summary.get('files_modified', [])
                    if files:
                        print(f"Files modified: {len(files)}")

                    cam_id = primer.get('cam_embedding_id', 'none')
                    print(f"CAM ID: {cam_id}")
                    print("-" * 50)

                except Exception as e:
                    print(f"Error reading {os.path.basename(primer_path)}: {e}")
                    print("-" * 50)

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
