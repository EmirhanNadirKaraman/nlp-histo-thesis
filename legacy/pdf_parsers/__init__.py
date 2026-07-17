"""
PDF Parsers Package (legacy / comparison parsers)

Historical PDF parsers retained for research comparisons only. The
production PDF pipeline lives in ``pipeline/stages/pdf_text_extraction/``
and uses Docling directly via ``PipelineRunner`` — these parsers are not
the database ingestion path.

Parsers in this package:
- MarkerParser, NougatParser, PDFFiguresParser, DoclingParser,
  PyMuPDF4LLMParser, EnsemblePDFParser

The ``BasePDFParser`` subclasses return the same hierarchical row format as the XML parser::

    {
        'path_list':   [...],   # section hierarchy as list
        'path_string': str,     # "Section > Subsection"
        'depth':       int,
        'text':        str,
    }
"""

from .base_parser import BasePDFParser

__all__ = ['BasePDFParser']

# Parsers are imported lazily to avoid dependency issues
try:
    from .marker_parser import MarkerParser  # noqa: F401
    __all__.append('MarkerParser')
except ImportError:
    pass

try:
    from .nougat_parser import NougatParser  # noqa: F401
    __all__.append('NougatParser')
except ImportError:
    pass

try:
    from .pdffigures_parser import PDFFiguresParser  # noqa: F401
    __all__.append('PDFFiguresParser')
except ImportError:
    pass

try:
    from .ensemble_parser import EnsemblePDFParser  # noqa: F401
    __all__.append('EnsemblePDFParser')
except ImportError:
    pass
