# terminology_project

Created by Cécile MACAIRE & Ludivine ROBERT

## Aim
Python script which extracts and annotates terms from a corpus of articles about "Text-to-Speech System".


## Context
The goal of this project is to develop a term identification system for a specific domain.
It has been realized under the _Terminology_ course, as part of the _Natural language processing_ masters' degree in _Université de Lorraine (Nancy)_.


## Instructions

### Requirements 

Pandas

Spacy

### Running process

The script to run is termsprocessing.py. The following instructions need to be filled out:
 * type of annotation:
   * 1 = terms annotation
   * 2 = terms annotation and IOB tags
   * 3 = terms annotation, IOB tags and Part-Of-Speech (POS) tags
 * input file that you want to annotate in _.txt_ format
 * name of your output file (also in _.txt_ format) 

Finally, enter the following command in your terminal:
```bash 
python termsprocessing.py -n 1  -i input.txt -o output.txt
```
