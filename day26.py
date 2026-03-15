import spacy
from spacy import displacy
import pandas as pd

# Load the English model
nlp = spacy.load('en_core_web_sm')

# Sample text for Named Entity Recognition
text = "Apple Inc. is looking at buying U.K. startup for $1 billion. The CEO, Tim Cook, will meet with the board next week."    

# Process the text with spaCy
doc = nlp(text)

# Extract and display named entities
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append({
            'Entity': ent.text,
            'Label': ent.label_,
            'Explanation': spacy.explain(ent.label_)    
        })
    return pd.DataFrame(entities)


entities_df = extract_entities(doc)

print("Extracted Named Entities:")
print(entities_df)


# Visualize the named entities in the text
displacy.render(doc, style='ent', jupyter=True)

# Save entities to a CSV file
entities_df.to_csv('extracted_entities.csv', index=False)
print("Named entities have been extracted and saved to 'extracted_entities.csv'.")