# information about spaCy can be found here: https://spacy.io/usage/spacy-101

# !pip3 install -U spacy
# !python3 -m spacy download en_core_web_sm
import spacy, sys

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

doc = nlp(text)

# Analyze syntax
noun_phrases=[]
verbs=[]

for chunk in doc.noun_chunks:
    noun_phrases.append(chunk.text)

for token in doc:
    if token.pos_ == 'VERB':
        verbs.append(token.lemma_)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

print(f"noun_phrases: {noun_phrases}")
print(f"verbs: {verbs}")

#for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#            token.shape_, token.is_alpha, token.is_stop)