# The model name as it is STORED in entities.model_name — spaCy's ``nlp.meta["name"]``,
# which omits the language prefix that the package name carries. Loading
# ``en_core_sci_lg`` yields meta lang="en" + name="core_sci_lg", and ner.py persists the
# latter. Consumers must filter on this value, not on the package name: defaulting to
# "en_core_sci_lg" matches zero rows and makes `ner merge` / `ner export` exit 0 having
# silently done nothing (B-115).
DEFAULT_MODEL_NAME = "core_sci_lg"

# The relevant TUIs for Diseases, Neoplasms, Mental Disorders, and Symptoms
# Mapping of TUI codes to their human-readable UMLS Semantic Type names
UMLS_DISEASE_TYPES = {
    'T047': 'Disease or Syndrome',
    'T191': 'Neoplastic Process',
    'T048': 'Mental or Behavioral Dysfunction',
    'T037': 'Injury or Poisoning',
    'T046': 'Pathologic Function',
    'T020': 'Congenital Abnormality',
    'T184': 'Sign or Symptom',
    'T033': 'Finding'
}