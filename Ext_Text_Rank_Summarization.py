import sys
import spacy
import pytextrank

nlp = spacy.load("en_core_web_lg")

nlp.add_pipe("textrank")

print("This is an example of Extractive Text Summarization using Text Rank Algorithm \n")
print("sys.stdin.read() method accepts a line as the input from the user until a special character like 'Enter Key' and followed by 'Ctrl + D' and then stores the input as the string.")
print("NOTE: Ctrl + D works as the stop signal. \n")
print("Please enter your text below: \n")
example_text = sys.stdin.read()

print('Original Document Size:',len(example_text))
doc = nlp(example_text)

for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
    print(sent)
    print('Summary Length:',len(sent))
