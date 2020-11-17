# terminology_project

## Aim

Python script which extract and annotate terms from a corpus build on articles about Multilingual Text-to-Speech system.

## Context

It has been realized under the _Terminology_ course, as part of the _Natural language processing_ masters' degree in _Université de Lorraine (Nancy)_.

## Instructions

There is three differents way to annonate your text.
First, you should choose the type of annotation:
1= terms annotation
2= terms annoation and IOB tags
3= terms annotaion, IOB tags and Part-Of-Speech (POS) tags

Then, you should have the text you want to process in a _.txt_ format.

Finally, enter the following command in your terminal:
  python termsprocessing.py -n 1  -i input.txt -o output.txt"
