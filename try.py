import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])
print("I can import")