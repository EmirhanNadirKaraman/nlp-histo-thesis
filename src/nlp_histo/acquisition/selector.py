from pathlib import Path
from typing import List, Set
import pandas as pd
import re


class HistopathologyFileSelector:
    """Filters large CSV files for histopathology-related papers efficiently using chunked processing."""

    def __init__(
        self,
        index_file_path: str = 'files/oa_file_list.csv',
        output_file_path: str = 'target_pmc_ids.txt',
        chunk_size: int = 100000
    ):
        """
        Initialize the file selector.

        Args:
            index_file_path: Path to the CSV file to process
            output_file_path: Path to save the filtered PMC IDs
            chunk_size: Number of rows to process per chunk
        """
        self.index_file_path = Path(index_file_path)
        self.output_file_path = Path(output_file_path)
        self.chunk_size = chunk_size

        # INCLUSION CRITERIA - Papers we want to find
        self.include_terms = [
            r"\bhistopathology\b", r"\bhistology\b", r"\bhistological\b",
            r"\bpathology\b", r"\bpathological\b",
            r"\bwhole slide image\b", r"\bwsi\b", r"\bdigital pathology\b",
            r"\bhematoxylin\b", r"\beosin\b", r"\bh&e\b",
            r"\bimmunohistochemistry\b", r"\bihc\b",
            r"\bbiopsy\b", r"\bresection\b", r"\bstained section\b",
            r"\bdermatopathology\b", r"\bneuropathology\b",
            r"\bcytopathology\b", r"\bsurgical pathology\b"
        ]

        # EXCLUSION CRITERIA - Papers we want to filter out
        self.exclude_terms = [
            r"\bmagnetic resonance\b", r"\bmri\b",
            r"\bcomputed tomography\b", r"\bct scan\b",
            r"\bultrasound\b", r"\bsonography\b",
            r"\bradiography\b", r"\bx-ray\b", r"\bxray\b",
            r"\bendoscopy\b", r"\bcolonoscopy\b", r"\blaparoscopy\b",
            r"\bwestern blot\b", r"\bflow cytometry\b", r"\bpcr\b", r"\brna-seq\b",
            r"\bveterinary\b", r"\banimal model\b", r"\bmouse\b", r"\brat\b", r"\bmurine\b"
        ]

        # Compile regex patterns once for efficiency
        self.include_pattern = re.compile('|'.join(self.include_terms), re.IGNORECASE)
        self.exclude_pattern = re.compile('|'.join(self.exclude_terms), re.IGNORECASE)

    def _validate_file(self) -> None:
        """Validate that the index file exists."""
        if not self.index_file_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")
        if not self.index_file_path.is_file():
            raise ValueError(f"Path is not a file: {self.index_file_path}")

    def _filter_chunk(self, chunk: pd.DataFrame) -> Set[str]:
        """
        Filter a chunk for relevant PMC IDs.

        Args:
            chunk: DataFrame chunk to filter

        Returns:
            Set of relevant PMC IDs found in this chunk
        """
        # Validate required columns exist
        required_cols = ['Article Citation', 'Accession ID']
        missing_cols = [col for col in required_cols if col not in chunk.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Use Article Citation for searching (contains title and journal info)
        search_text = chunk['Article Citation'].fillna('')

        # Apply inclusion and exclusion filters
        mask_include = search_text.str.contains(self.include_pattern, na=False)
        mask_exclude = search_text.str.contains(self.exclude_pattern, na=False)

        # Final filter: include AND NOT exclude
        final_mask = mask_include & (~mask_exclude)

        # Extract unique PMC IDs
        relevant_papers = chunk[final_mask]
        return set(relevant_papers['Accession ID'].dropna().unique())

    def process_file(self) -> List[str]:
        """
        Process the entire CSV file in chunks and return filtered PMC IDs.

        Returns:
            List of unique PMC IDs that match the criteria

        Raises:
            FileNotFoundError: If the index file doesn't exist
            ValueError: If required columns are missing
        """
        self._validate_file()

        print(f"Loading CSV in chunks from: {self.index_file_path}")
        print(f"Chunk size: {self.chunk_size:,} rows")

        target_pmcids: Set[str] = set()
        total_rows = 0
        chunk_count = 0

        try:
            # Process file in chunks to avoid memory issues
            chunks = pd.read_csv(
                self.index_file_path,
                chunksize=self.chunk_size,
                on_bad_lines='skip',
                dtype=str
            )

            for chunk in chunks:
                chunk_count += 1
                total_rows += len(chunk)

                # Filter this chunk
                chunk_pmcids = self._filter_chunk(chunk)
                target_pmcids.update(chunk_pmcids)

                print(f"Processed chunk {chunk_count}: {total_rows:,} total rows, "
                      f"{len(target_pmcids):,} relevant papers so far")

        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {self.index_file_path} is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")

        print("\nFiltering complete.")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Relevant histopathology papers found: {len(target_pmcids):,}")

        return sorted(target_pmcids)

    def save_results(self, pmcids: List[str]) -> None:
        """
        Save the filtered PMC IDs to a text file.

        Args:
            pmcids: List of PMC IDs to save
        """
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file_path, 'w') as f:
            for pmcid in pmcids:
                f.write(f"{pmcid}\n")

        print(f"Saved {len(pmcids):,} PMC IDs to: {self.output_file_path}")

    def run(self) -> List[str]:
        """
        Run the complete filtering process and save results.

        Returns:
            List of filtered PMC IDs
        """
        try:
            pmcids = self.process_file()
            self.save_results(pmcids)
            return pmcids
        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main entry point for the file selector script."""
    selector = HistopathologyFileSelector()
    selector.run()


if __name__ == '__main__':
    main()
