"""Pipeline output-handler implementations."""
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.writer import TextFileWriter
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.db_ingester import PostgresDatabaseIngester
from nlp_histo.pipeline.stages.pdf_text_extraction.outputs.media_json_writer import MediaJsonWriter

__all__ = ["TextFileWriter", "PostgresDatabaseIngester", "MediaJsonWriter"]
