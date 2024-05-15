import spacy
from spacy.matcher import Matcher
from pdf_reader import generate_summary

nlp = spacy.load('en_core_web_sm')

entity_matcher = Matcher(nlp.vocab)

entity_pattern = [
    [{"LOWER": "classification"}, {"LOWER": "binary"}],
    [{"LOWER": "ide"}, {"LOWER": "carbon"}]
    ]
# Add entity patters to the Matcher
for pattern in entity_pattern:
    entity_matcher.add("Entity", [pattern])

chunks = generate_summary(pdf_path="./pdfs/")
# Process each text and extract entities
for chunk in chunks:
    doc = nlp(chunk)
    matches = entity_matcher(doc)

    # Extract matched entities
    matched_entities = [doc[start:end].text for match_id, start, end in matches]

    # print matched entities for the current chunk
    if matched_entities:
        print("Matched entities in the chunk: ", matched_entities)
    else:
        print("No matches found")
