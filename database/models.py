"""
SQLAlchemy ORM models for the NLP-histo PostgreSQL database: the source paper
(Document), its hierarchical text (TextElement), extracted media (Figure,
Table), NER entities (Entity), and the summarization-pipeline persistence
tables further down the file.
"""

from sqlalchemy import (
    Boolean, Column, Integer, String, Text, TIMESTAMP,
    ForeignKey, ARRAY, Index, Float, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Document(Base):
    """
    One source paper (PMC article), keyed by ``pmcid``. Holds either an XML- or
    PDF-derived body (see ``text_source``) and is the cascade-delete root for its
    text elements, figures, and tables.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pmcid = Column(String(50), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    title = Column(Text)
    journal = Column(String(255))
    publication_year = Column(Integer)
    text_source = Column(String(20), default='xml')  # 'xml' or 'pdf'
    processed_at = Column(TIMESTAMP, default=func.now())
    created_at = Column(TIMESTAMP, default=func.now())
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())

    # Relationship to text elements
    text_elements = relationship(
        "TextElement",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    # Relationship to figures
    figures = relationship(
        "Figure",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    # Relationship to tables
    tables = relationship(
        "Table",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(pmcid='{self.pmcid}', title='{self.title[:50] if self.title else None}...')>"


class TextElement(Base):
    """
    Represents a single text element (paragraph) with its hierarchical path.
    Each text element belongs to one document.
    """
    __tablename__ = 'text_elements'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)

    # Unique path for this text element
    unique_path = Column(Text, unique=True, nullable=False, index=True)

    # Hierarchical information
    path_list = Column(ARRAY(Text), nullable=False)  # PostgreSQL array
    path_string = Column(Text, nullable=False, index=True)
    depth = Column(Integer, nullable=False, index=True)
    position_in_section = Column(Integer, nullable=False)

    # The actual text content
    text_content = Column(Text, nullable=False)

    # Figure/table references mentioned in this text
    # Format: {'figures': ['1', '2'], 'tables': ['3']}
    references = Column(JSON)

    # Metadata
    word_count = Column(Integer)
    char_count = Column(Integer)
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="text_elements")

    # Relationships to figure/table references (many-to-many)
    figure_references = relationship("TextElementFigureReference", back_populates="text_element")
    table_references = relationship("TextElementTableReference", back_populates="text_element")

    # Add a composite unique constraint
    __table_args__ = (
        Index('idx_document_path_position', 'document_id', 'path_string', 'position_in_section', unique=True),
    )

    def __repr__(self):
        text_preview = self.text_content[:50] if self.text_content else ""
        return f"<TextElement(unique_path='{self.unique_path}', text='{text_preview}...')>"

    @property
    def section_hierarchy(self):
        """Returns the full hierarchy as a formatted string."""
        return " > ".join(self.path_list) if self.path_list else "Root"

    def calculate_metadata(self):
        """Calculate and update word_count and char_count."""
        if self.text_content:
            self.char_count = len(self.text_content)
            self.word_count = len(self.text_content.split())


# Junction table for text_element ↔ figure references (many-to-many)
class TextElementFigureReference(Base):
    """Links text elements to figures they mention."""
    __tablename__ = 'text_element_figure_references'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_element_id = Column(Integer, ForeignKey('text_elements.id', ondelete='CASCADE'), nullable=False, index=True)
    figure_id = Column(Integer, ForeignKey('figures.id', ondelete='CASCADE'), nullable=False, index=True)

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationships
    text_element = relationship("TextElement", back_populates="figure_references")
    figure = relationship("Figure", back_populates="text_references")

    __table_args__ = (
        Index('idx_text_element_figure', 'text_element_id', 'figure_id', unique=True),
    )


# Junction table for text_element ↔ table references (many-to-many)
class TextElementTableReference(Base):
    """Links text elements to tables they mention."""
    __tablename__ = 'text_element_table_references'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_element_id = Column(Integer, ForeignKey('text_elements.id', ondelete='CASCADE'), nullable=False, index=True)
    table_id = Column(Integer, ForeignKey('tables.id', ondelete='CASCADE'), nullable=False, index=True)

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationships
    text_element = relationship("TextElement", back_populates="table_references")
    table = relationship("Table", back_populates="text_references")

    __table_args__ = (
        Index('idx_text_element_table', 'text_element_id', 'table_id', unique=True),
    )


class Figure(Base):
    """
    A figure cropped from the source PDF: its caption, the stored image file, and
    the page/bbox it was cut from. ``text_references`` links the paragraphs that
    mention it.
    """
    __tablename__ = 'figures'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)

    # Figure identification
    figure_id = Column(String(100))  # e.g., "fig1", "fig2a"
    figure_label = Column(String(50))  # e.g., "Figure 1", "Fig 2A"
    figure_number = Column(String(20))  # Just the number: "1", "2A", etc.

    # Caption and context
    caption_text = Column(Text)

    # Source location in the PDF (Docling coords, y=0 at page bottom ⇒ y1>y2).
    # Added 0014 (B-075): mirror Table; populated by the cropper via CroppedMedia.
    page_number = Column(Integer)  # Page where figure appears
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)

    # Image file information
    graphic_ref = Column(String(255))  # Original filename from XML
    image_filename = Column(String(255))  # Stored filename
    image_path = Column(Text)  # Full path to stored image
    image_format = Column(String(20))  # jpg, png, tif, etc.

    # Location in document
    section_context = Column(Text)  # Which section this figure appears in

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="figures")

    # Relationship to text element references (which texts mention this figure)
    text_references = relationship("TextElementFigureReference", back_populates="figure")

    # Add a composite unique constraint
    __table_args__ = (
        Index('idx_document_figure_id', 'document_id', 'figure_id', unique=True),
    )

    def __repr__(self):
        return f"<Figure(figure_id='{self.figure_id}', label='{self.figure_label}')>"


class Table(Base):
    """
    A table extracted from the source paper: caption, optional raw content, and
    the cropped table image. ``text_references`` links the paragraphs that
    mention it.
    """
    __tablename__ = 'tables'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True)

    # Table identification
    table_id = Column(String(100))  # e.g., "table1", "table2"
    table_label = Column(String(50))  # e.g., "Table 1", "Table II"
    table_number = Column(String(20))  # Just the number: "1", "2", "II", etc.

    # Caption and content
    caption_text = Column(Text)
    table_content = Column(Text)  # Raw table content (if extracted)

    # Image file information (extracted table image)
    image_filename = Column(String(255))  # Stored filename
    image_path = Column(Text)  # Full path to stored image

    # Location in document
    section_context = Column(Text)  # Which section this table appears in
    page_number = Column(Integer)  # Page where table appears

    # Bounding box (if from PDFFigures2)
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to document
    document = relationship("Document", back_populates="tables")

    # Relationship to text element references (which texts mention this table)
    text_references = relationship("TextElementTableReference", back_populates="table")

    # Add a composite unique constraint
    __table_args__ = (
        Index('idx_document_table_id', 'document_id', 'table_id', unique=True),
    )

    def __repr__(self):
        return f"<Table(table_id='{self.table_id}', label='{self.table_label}')>"


class Entity(Base):
    """
    A named entity found in one TextElement by the scispaCy NER pass, with its
    optional UMLS link (CUI, canonical name, semantic types) from the UMLS linker.
    """
    __tablename__ = 'entities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_element_id = Column(Integer, ForeignKey('text_elements.id', ondelete='CASCADE'), nullable=False, index=True)

    # Entity information
    entity_text = Column(Text, nullable=False)
    entity_label = Column(String(50), nullable=False, index=True)  # PERSON, ORG, GPE, etc.

    # Position in text
    start_char = Column(Integer)
    end_char = Column(Integer)

    # UMLS entity linking (from scispacy)
    umls_cui = Column(String(20), index=True)  # UMLS Concept Unique Identifier
    umls_score = Column(Float)  # Linking confidence score (0.0-1.0)
    canonical_name = Column(Text)  # Human-readable UMLS concept name
    semantic_types = Column(ARRAY(Text), nullable=True)  # To store TUI codes or names

    # NER model metadata
    model_name = Column(String(100))  # e.g., 'en_core_sci_sm', 'en_core_web_sm'
    linker_name = Column(String(50))  # e.g., 'umls', 'mesh', 'rxnorm'

    # Metadata
    created_at = Column(TIMESTAMP, default=func.now())

    # Relationship to text element
    text_element = relationship("TextElement", backref="entities")

    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_text_element_entity', 'text_element_id', 'entity_label'),
        Index('idx_entity_text', 'entity_text'),
        Index('idx_umls_cui', 'umls_cui'),
    )

    def __repr__(self):
        cui_info = f", CUI={self.umls_cui}" if self.umls_cui else ""
        canonical_info = f", canonical='{self.canonical_name}'" if self.canonical_name else ""
        return f"<Entity(text='{self.entity_text}', label='{self.entity_label}'{cui_info}{canonical_info})>"


class PipelineRun(Base):
    """
    Tracks one execution of SummarizationRunner.process() for a single paper.

    Every stage-level persistence table (added in later phases) will FK here,
    making this the root of all lineage tracing.
    """
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Human-readable unique key: "{pmcid}_{YYYYMMDDTHHmmss}"
    run_id = Column(String(100), unique=True, nullable=False, index=True)

    pmcid = Column(String(50), nullable=False, index=True)

    # Nullable: document may not be in the DB when the run starts
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # running / success / failed
    status = Column(String(20), nullable=False, default="running")

    # Snapshot of SummarizationRunner config (models, thresholds, etc.)
    config_snapshot = Column(JSON, nullable=True)

    # Set on failure
    error = Column(Text, nullable=True)

    # Narrative summary produced by the REDUCE stage
    narrative_summary = Column(Text, nullable=True)

    started_at = Column(TIMESTAMP, nullable=False, default=func.now())
    finished_at = Column(TIMESTAMP, nullable=True)

    document = relationship("Document", backref="pipeline_runs")

    __table_args__ = (
        Index("ix_pipeline_runs_pmcid_started_at", "pmcid", "started_at"),
    )

    def __repr__(self):
        return f"<PipelineRun(run_id='{self.run_id}', status='{self.status}')>"


# ── Phase 2: MAP findings & Normalized findings ────────────────────────────────

class SumMapFinding(Base):
    """
    One row per Finding that survived the grounding filter, stored after
    grounding scores are written in-place (post-scoring).
    Entities are PRE-normalization — use SumNormalFinding to see resolved names.
    """
    __tablename__ = "sum_map_findings"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id         = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid                   = Column(String(50),  nullable=False)
    chunk_id                = Column(String(20),  nullable=False)   # "C1", "C2", …
    position_in_chunk       = Column(Integer,     nullable=False)   # 0-indexed within chunk
    category                = Column(String(50),  nullable=False)
    claim                   = Column(Text,        nullable=False)
    confidence              = Column(String(20),  nullable=False)
    verbatim_support        = Column(Text,        nullable=False)
    subject_entity          = Column(Text,        nullable=True)    # pre-normalization
    outcome_entity          = Column(Text,        nullable=True)    # pre-normalization
    relation_type           = Column(String(50),  nullable=False)
    direction               = Column(String(20),  nullable=True)
    raw_relation_type       = Column(Text,        nullable=True)   # pre-coercion LLM string
    raw_direction           = Column(Text,        nullable=True)   # pre-coercion LLM string
    raw_category            = Column(Text,        nullable=True)   # pre-alias-repair LLM string
    grounding_score         = Column(Float,       nullable=True)
    evidence_refs           = Column(ARRAY(Text), nullable=True)
    scope_disease_subtype   = Column(Text,    nullable=True)
    scope_cohort_n          = Column(Integer, nullable=True)
    scope_assay_method      = Column(Text,    nullable=True)
    scope_biomarker_cutoff  = Column(Text,    nullable=True)
    scope_tissue_site       = Column(Text,    nullable=True)
    scope_treatment_context = Column(Text,    nullable=True)
    scope_endpoint          = Column(Text,    nullable=True)
    scope_study_design      = Column(Text,    nullable=True)
    created_at              = Column(TIMESTAMP, server_default="now()")

    __table_args__ = (
        Index("ix_smf_run",     "pipeline_run_id"),
        Index("ix_smf_pmcid",   "pmcid"),
        Index("ix_smf_subject", "subject_entity"),
        UniqueConstraint("pipeline_run_id", "chunk_id", "position_in_chunk",
                         name="uq_sum_map_finding_pos"),
    )


class SumMapVoterOutput(Base):
    """
    One row per (chunk, cascade level, voter) — captures the AuditableSummary
    each individual voter produced, including voters whose output was
    eventually discarded by agreement. Lets thesis analysis answer "what did
    each model say for chunk C5?" rather than only "which voter won".

    ``raw_output`` is the full AuditableSummary serialised as JSONB (Pydantic
    ``model_dump()``). ``failed = True`` rows carry ``raw_output = NULL`` and
    the API error in ``error_message``.

    ``is_selected`` marks the voter whose output became the per-chunk MAP
    result at this level. Exactly one row per (chunk, level) has
    is_selected=True when the level produced a result; zero when the level
    escalated and a higher tier produced the final.
    """
    __tablename__ = "sum_map_voter_outputs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid           = Column(String(50),  nullable=False)
    chunk_id        = Column(String(20),  nullable=False)   # "C1", "C2", …
    level           = Column(String(4),   nullable=False)   # "l1" | "l2" | "l3"
    voter_index     = Column(Integer,     nullable=False)   # 0-based within the level
    provider        = Column(String(20),  nullable=False)   # "openai" | "gemini" | "anthropic"
    model           = Column(String(100), nullable=False)
    is_selected     = Column(Boolean,     nullable=False)   # winner at this level (one per (chunk, level) at most)
    failed          = Column(Boolean,     nullable=False)   # API call raised
    error_message   = Column(Text,        nullable=True)
    finding_count   = Column(Integer,     nullable=False)   # 0 if failed
    latency_ms      = Column(Float,       nullable=True)
    raw_output      = Column(JSONB,       nullable=True)    # AuditableSummary.model_dump()
    created_at      = Column(TIMESTAMP,   server_default="now()")

    __table_args__ = (
        Index("ix_smvo_run",   "pipeline_run_id"),
        Index("ix_smvo_pmcid", "pmcid"),
        Index("ix_smvo_chunk", "pipeline_run_id", "pmcid", "chunk_id"),
        UniqueConstraint(
            "pipeline_run_id", "pmcid", "chunk_id", "level", "voter_index",
            name="uq_sum_map_voter_output",
        ),
    )


class SumNormalFinding(Base):
    """
    One row per NormalFinding output from the NORMALIZE stage.
    Entities are POST-normalization (e.g. CAEN -> CEAN after synonym lookup).
    """
    __tablename__ = "sum_normal_findings"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id         = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid                   = Column(String(50),  nullable=False)
    normal_id               = Column(String(100), nullable=False)
    subject_entity          = Column(Text,        nullable=True)
    outcome_entity          = Column(Text,        nullable=True)
    relation_type           = Column(String(50),  nullable=False)
    direction               = Column(String(20),  nullable=True)
    category                = Column(String(50),  nullable=False)
    predicate_text          = Column(Text,        nullable=False)
    mean_grounding_score    = Column(Float,       nullable=True)
    pmcids                  = Column(ARRAY(Text), nullable=True)
    scope_disease_subtype   = Column(Text,    nullable=True)
    scope_cohort_n          = Column(Integer, nullable=True)
    scope_assay_method      = Column(Text,    nullable=True)
    scope_biomarker_cutoff  = Column(Text,    nullable=True)
    scope_tissue_site       = Column(Text,    nullable=True)
    scope_treatment_context = Column(Text,    nullable=True)
    scope_endpoint          = Column(Text,    nullable=True)
    scope_study_design      = Column(Text,    nullable=True)
    created_at              = Column(TIMESTAMP, server_default="now()")

    spans = relationship(
        "SumNormalFindingSpan",
        back_populates="normal_finding",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_snf_run",     "pipeline_run_id"),
        Index("ix_snf_pmcid",   "pmcid"),
        Index("ix_snf_subject", "subject_entity"),
        Index("ix_snf_outcome", "outcome_entity"),
        UniqueConstraint("pipeline_run_id", "normal_id", name="uq_sum_normal_finding"),
    )


class SumNormalFindingSpan(Base):
    """
    One row per SourceSpan attached to a SumNormalFinding.
    Links to the originating TextElement for full provenance.
    """
    __tablename__ = "sum_normal_finding_spans"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    normal_finding_id   = Column(Integer, ForeignKey("sum_normal_findings.id", ondelete="CASCADE"), nullable=False)
    sentence_id         = Column(String(20), nullable=False)
    pmcid               = Column(String(50), nullable=False)
    text_element_id     = Column(Integer, ForeignKey("text_elements.id", ondelete="SET NULL"), nullable=True)
    verbatim            = Column(Text, nullable=False)
    created_at          = Column(TIMESTAMP, server_default="now()")

    normal_finding = relationship("SumNormalFinding", back_populates="spans")

    __table_args__ = (
        Index("ix_snfs_finding", "normal_finding_id"),
        Index("ix_snfs_te",      "text_element_id"),
        UniqueConstraint("normal_finding_id", "sentence_id", name="uq_sum_nf_span"),
    )


# ── Phase 3: Finding groups ────────────────────────────────────────────────────

class SumFindingGroup(Base):
    """
    One row per FindingGroup produced by the GROUP stage.

    All NormalFindings with the same (subject, outcome, relation_type) land
    in the same group regardless of direction — contradictions are handled by
    the RELATE stage (Phase 5), not here.
    """
    __tablename__ = "sum_finding_groups"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id     = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid               = Column(String(50),  nullable=False)
    group_id            = Column(String(100), nullable=False)   # "GRP_{sha8}_{sha8}_{relation_type}"
    subject_entity      = Column(Text,        nullable=False)
    outcome_entity      = Column(Text,        nullable=False)
    relation_type       = Column(String(50),  nullable=False)
    category            = Column(String(50),  nullable=False)
    scope_heterogeneity = Column(Float,       nullable=False)
    direction_counts    = Column(JSON,        nullable=False)    # {"positive": 2, "negative": 1, …}
    created_at          = Column(TIMESTAMP,   server_default="now()")

    members = relationship(
        "SumGroupMember",
        back_populates="group",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sfg_run",     "pipeline_run_id"),
        Index("ix_sfg_pmcid",   "pmcid"),
        Index("ix_sfg_subject", "subject_entity"),
        Index("ix_sfg_outcome", "outcome_entity"),
        UniqueConstraint("pipeline_run_id", "group_id", name="uq_sum_finding_group"),
    )


class SumGroupMember(Base):
    """
    Junction table: one row per (FindingGroup, NormalFinding) membership.

    normal_finding_id is nullable — it stays NULL when Phase 2 persistence
    was skipped (db=None) or failed.  normal_id is always stored as a string
    so membership can still be traced without the FK.
    """
    __tablename__ = "sum_group_members"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    finding_group_id    = Column(Integer, ForeignKey("sum_finding_groups.id", ondelete="CASCADE"), nullable=False)
    normal_finding_id   = Column(Integer, ForeignKey("sum_normal_findings.id", ondelete="SET NULL"), nullable=True)
    normal_id           = Column(String(100), nullable=False)   # string backup always present
    created_at          = Column(TIMESTAMP, server_default="now()")

    group          = relationship("SumFindingGroup", back_populates="members")
    normal_finding = relationship("SumNormalFinding")

    __table_args__ = (
        Index("ix_sgm_group",  "finding_group_id"),
        Index("ix_sgm_nf",     "normal_finding_id"),
        UniqueConstraint("finding_group_id", "normal_id", name="uq_sum_group_member"),
    )


# ── Phase 4: Canonical rules ──────────────────────────────────────────────────

class SumCanonicalRule(Base):
    __tablename__ = "sum_canonical_rules"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id      = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid                = Column(String(50),  nullable=False)
    canonical_id         = Column(String(100), nullable=False)
    finding_group_id     = Column(Integer, ForeignKey("sum_finding_groups.id", ondelete="SET NULL"), nullable=True)
    group_id             = Column(String(100), nullable=False)
    subject_entity       = Column(Text,        nullable=False)
    outcome_entity       = Column(Text,        nullable=False)
    relation_type        = Column(String(50),  nullable=False)
    direction            = Column(String(20),  nullable=True)
    predicate_text       = Column(Text,        nullable=False)
    is_conflicted        = Column(Boolean,      nullable=False, default=False)
    study_coverage       = Column(String(20),  nullable=False)
    category             = Column(String(50),  nullable=False)
    pmcids               = Column(ARRAY(Text), nullable=True)
    member_normal_ids    = Column(ARRAY(Text), nullable=True)
    mean_grounding_score = Column(Float,       nullable=True)
    finding_count        = Column(Integer,     nullable=False)
    subject_cui          = Column(String(20),  nullable=True)   # UMLS CUI for subject_entity
    outcome_cui          = Column(String(20),  nullable=True)   # UMLS CUI for outcome_entity
    created_at           = Column(TIMESTAMP,   server_default="now()")

    __table_args__ = (
        Index("ix_scr_run",          "pipeline_run_id"),
        Index("ix_scr_pmcid",        "pmcid"),
        Index("ix_scr_subject",      "subject_entity"),
        Index("ix_scr_outcome",      "outcome_entity"),
        Index("ix_scr_subject_cui",  "subject_cui"),
        Index("ix_scr_outcome_cui",  "outcome_cui"),
        UniqueConstraint("pipeline_run_id", "canonical_id", name="uq_sum_canonical_rule"),
    )


# ── Phase 5: Relations ────────────────────────────────────────────────────────

class SumRelation(Base):
    __tablename__ = "sum_relations"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id      = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid                = Column(String(50),  nullable=False)
    rule_id_a            = Column(String(100), nullable=False)
    rule_id_b            = Column(String(100), nullable=False)
    canonical_rule_a_id  = Column(Integer, ForeignKey("sum_canonical_rules.id", ondelete="CASCADE"), nullable=True)
    canonical_rule_b_id  = Column(Integer, ForeignKey("sum_canonical_rules.id", ondelete="CASCADE"), nullable=True)
    relation_type        = Column(String(20),  nullable=False)
    nli_score_a_to_b     = Column(Float,       nullable=False)
    nli_score_b_to_a     = Column(Float,       nullable=False)
    created_at           = Column(TIMESTAMP,   server_default="now()")

    __table_args__ = (
        Index("ix_sr_run",    "pipeline_run_id"),
        Index("ix_sr_pmcid",  "pmcid"),
        Index("ix_sr_rule_a", "rule_id_a"),
        Index("ix_sr_rule_b", "rule_id_b"),
    )


# ── Phase 6: Final rules ──────────────────────────────────────────────────────

class SumFinalRule(Base):
    __tablename__ = "sum_final_rules"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id      = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    pmcid                = Column(String(50),  nullable=False)
    final_id             = Column(String(100), nullable=False)
    canonical_rule_id    = Column(Integer, ForeignKey("sum_canonical_rules.id", ondelete="SET NULL"), nullable=True)
    canonical_id         = Column(String(100), nullable=False)
    
    subject_entity       = Column(Text,        nullable=False)
    outcome_entity       = Column(Text,        nullable=False)
    relation_type        = Column(String(50),  nullable=False)
    direction            = Column(String(20),  nullable=True)
    predicate_text       = Column(Text,        nullable=False)
    category             = Column(String(50),  nullable=False)
    
    final_score          = Column(Float,       nullable=False)
    support_count        = Column(Integer,     nullable=False)
    contradict_count     = Column(Integer,     nullable=False)
    # Always 0 — SCOPE_QUALIFY classifier branch was never implemented (B-006);
    # column retained so existing reads (HTML inspector, downstream consumers)
    # don't break. Drop via Alembic if the branch is not reinstated.
    scope_qualify_count  = Column(Integer,     nullable=False)
    is_contradicted      = Column(Boolean,     nullable=False)
    contradicted_by      = Column(ARRAY(Text), nullable=True)
    # Which RESOLVE branch produced final_score. See models.FinalRule.score_mode.
    # Nullable for backward compatibility with rows written before the column existed.
    score_mode           = Column(String(30),  nullable=True)
    created_at           = Column(TIMESTAMP,   server_default="now()")

    __table_args__ = (
        Index("ix_sfr_run",     "pipeline_run_id"),
        Index("ix_sfr_pmcid",   "pmcid"),
        Index("ix_sfr_subject", "subject_entity"),
        Index("ix_sfr_outcome", "outcome_entity"),
        UniqueConstraint("pipeline_run_id", "final_id", name="uq_sum_final_rule"),
    )


# ── Rejection summary ────────────────────────────────────────────────────────

class SumRejectionSummary(Base):
    """
    Aggregate rejection statistics for one pipeline run.

    One row per pipeline run.  Covers two rejection points:
      1. Grounding filter (MAP) — findings dropped because NLI entailment < threshold.
      2. GROUP exclusion — NormalFindings excluded because subject/outcome entity is
         missing or relation_type is unclear.

    Detail rows live in sum_rejected_findings (FK sum_rejection_summary_id).
    """
    __tablename__ = "sum_rejection_summaries"

    id                           = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id              = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, unique=True)
    pmcid                        = Column(String(50),  nullable=False)
    grounding_threshold          = Column(Float,       nullable=True)
    map_findings_total           = Column(Integer,     nullable=False)
    map_grounding_rejected       = Column(Integer,     nullable=False)
    normal_findings_total        = Column(Integer,     nullable=False)
    non_groupable_total          = Column(Integer,     nullable=False)
    non_groupable_no_subject     = Column(Integer,     nullable=False)
    non_groupable_no_outcome     = Column(Integer,     nullable=False)
    non_groupable_unclear_relation = Column(Integer,   nullable=False)
    grounding_rejection_rate     = Column(Float,       nullable=False)
    non_groupable_rate           = Column(Float,       nullable=False)
    grounding_rejected_by_category = Column(JSON,     nullable=True)
    created_at                   = Column(TIMESTAMP,   server_default="now()")

    rejected_findings = relationship(
        "SumRejectedFinding",
        back_populates="rejection_summary",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_srs_run",   "pipeline_run_id"),
        Index("ix_srs_pmcid", "pmcid"),
    )


class SumRejectedFinding(Base):
    """
    One row per finding dropped or excluded during a pipeline run.

    stage values:
      "grounding_map"      — dropped by the NLI grounding filter after MAP.
      "group_non_groupable"— excluded from GROUP because subject/outcome entity
                             is None or relation_type is unclear.
    """
    __tablename__ = "sum_rejected_findings"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id        = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    rejection_summary_id   = Column(Integer, ForeignKey("sum_rejection_summaries.id", ondelete="CASCADE"), nullable=False)
    pmcid                  = Column(String(50),  nullable=False)
    stage                  = Column(String(30),  nullable=False)   # grounding_map | group_non_groupable
    reason                 = Column(Text,        nullable=False)
    claim                  = Column(Text,        nullable=False)
    category               = Column(String(50),  nullable=False)
    chunk_id               = Column(String(20),  nullable=True)    # set for grounding_map only
    grounding_score        = Column(Float,       nullable=True)
    subject_entity         = Column(Text,        nullable=True)
    outcome_entity         = Column(Text,        nullable=True)
    relation_type          = Column(String(50),  nullable=True)
    created_at             = Column(TIMESTAMP,   server_default="now()")

    rejection_summary = relationship("SumRejectionSummary", back_populates="rejected_findings")

    __table_args__ = (
        Index("ix_srf_run",      "pipeline_run_id"),
        Index("ix_srf_summary",  "rejection_summary_id"),
        Index("ix_srf_pmcid",    "pmcid"),
        Index("ix_srf_stage",    "stage"),
        Index("ix_srf_category", "category"),
    )


# ── Corpus-level relations (cross-paper / intra-paper graph) ──────────────────

class SumCorpusRelation(Base):
    """
    One row per relation produced by CorpusRelateStage.

    The table is replaced wholesale on each CorpusRelateStage run — there is
    no per-paper FK since corpus relations span all papers.  corpus_run_id is
    a YYYYMMDDTHHmmss timestamp string that identifies which run wrote the row.

    comparison_scope: "cross_paper" | "intra_paper"
    relation_type:    "SUPPORT" | "CONTRADICT"  (SCOPE_QUALIFY removed — B-006)
    """
    __tablename__ = "sum_corpus_relations"

    id                   = Column(Integer,     primary_key=True, autoincrement=True)
    corpus_run_id        = Column(String(20),  nullable=False, index=True)
    relation_id          = Column(String(20),  nullable=False)      # COR_{sha8}
    comparison_scope     = Column(String(20),  nullable=False)      # cross_paper | intra_paper
    same_paper           = Column(Boolean,     nullable=False)
    rule_id_a            = Column(String(100), nullable=False)
    rule_id_b            = Column(String(100), nullable=False)
    pmcid_a              = Column(String(50),  nullable=False)
    pmcid_b              = Column(String(50),  nullable=False)
    relation_type        = Column(String(20),  nullable=False)
    nli_score_a_to_b     = Column(Float,       nullable=False)
    nli_score_b_to_a     = Column(Float,       nullable=False)
    subject_entity       = Column(Text,        nullable=False)
    outcome_entity       = Column(Text,        nullable=False)
    category             = Column(String(50),  nullable=False)
    relation_type_structural = Column(String(50), nullable=False)
    direction_a          = Column(String(20),  nullable=True)
    direction_b          = Column(String(20),  nullable=True)
    predicate_a          = Column(Text,        nullable=False)
    predicate_b          = Column(Text,        nullable=False)
    mean_grounding_a     = Column(Float,       nullable=True)
    mean_grounding_b     = Column(Float,       nullable=True)
    finding_count_a      = Column(Integer,     nullable=False)
    finding_count_b      = Column(Integer,     nullable=False)
    supporting_pmcids_a  = Column(ARRAY(Text), nullable=True)
    supporting_pmcids_b  = Column(ARRAY(Text), nullable=True)
    is_conflicted_a      = Column(Boolean,      nullable=False, default=False)
    study_coverage_a     = Column(String(20),  nullable=False)
    is_conflicted_b      = Column(Boolean,      nullable=False, default=False)
    study_coverage_b     = Column(String(20),  nullable=False)
    scope_check_result   = Column(String(30),  nullable=False, default="scope_unknown")
    scope_note           = Column(Text,        nullable=True)
    created_at           = Column(TIMESTAMP,   server_default="now()")

    __table_args__ = (
        Index("ix_screl_run",       "corpus_run_id"),
        Index("ix_screl_scope",     "comparison_scope"),
        Index("ix_screl_pmcid_a",   "pmcid_a"),
        Index("ix_screl_pmcid_b",   "pmcid_b"),
        Index("ix_screl_rule_a",    "rule_id_a"),
        Index("ix_screl_rule_b",    "rule_id_b"),
        Index("ix_screl_type",      "relation_type"),
        UniqueConstraint("corpus_run_id", "relation_id", name="uq_sum_corpus_relation"),
    )


# ── LLM judge evaluation cache ────────────────────────────────────────────────

class LlmJudgeCache(Base):
    """
    Postgres-backed cache for LLM judge evaluation results.

    Each row stores one judge call: the request context, the Opus result,
    and enough metadata to support targeted cache invalidation (by model,
    prompt version, schema version, task, or paper).

    Cache key is content-addressed:
        sha256(model + prompt_version + schema_version + task + normalized_request_payload)
    """
    __tablename__ = "llm_judge_cache"

    cache_key       = Column(String(64),  primary_key=True)
    task            = Column(String(50),  nullable=False)
    judge_model     = Column(String(100), nullable=False)
    prompt_version  = Column(String(20),  nullable=False)
    schema_version  = Column(String(20),  nullable=False)
    label_source    = Column(String(50),  nullable=False)
    request_payload = Column(JSONB,       nullable=False)
    result_payload  = Column(JSONB,       nullable=False)
    pmcid           = Column(String(50),  nullable=True)
    pipeline_run_id = Column(Integer,     nullable=True)
    source_id       = Column(Integer,     nullable=True)
    created_at      = Column(TIMESTAMP,   server_default="now()")
    updated_at      = Column(TIMESTAMP,   server_default="now()", onupdate=func.now())

    __table_args__ = (
        Index("ix_ljc_task",       "task"),
        Index("ix_ljc_versions",   "judge_model", "prompt_version", "schema_version"),
        Index("ix_ljc_pmcid_run",  "pmcid", "pipeline_run_id"),
    )
