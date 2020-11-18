#!/usr/bin/env python
# coding: utf-8

### Terminology - Project
# Authors: Cécile MACAIRE & Ludivine ROBERT 

### Librairies
from os import listdir
from os.path import isfile, join

import pandas as pd
import spacy
import string as s
import argparse
import sys

spacy_nlp = spacy.load('en_core_web_sm')

rule_adj = ['accented', 'acoustic', 'artificial', 'attentional', 'autoregressive', 'bidirectional', 'bilingual',
            'continuous', 'cross-lingual', 'emotional', 'fluent', 'gated', 'generated', 'intelligible', 'labelled',
            'modern', 'monolingual', 'multilingual', 'multispeaker', 'neural', 'phonetic', 'substantial', 'supervised',
            'target', 'training', 'unlabelled', 'untranscribed', 'unsupervised', 'vanilla']

rule_4 = ['accent', 'accuracy', 'activation', 'adaptation', 'algorithm', 'aligner', 'alignment', 'approach',
          'architecture', 'attention', 'attribute', 'bank', 'boundary', 'category', 'cell', 'class', 'classifier',
          'cluster', 'clustering', 'coefficient', 'component', 'concatenation', 'content', 'context', 'contour',
          'control', 'conversion', 'corpora', 'coverage', 'decoding', 'detection', 'detection', 'device', 'dictionary',
          'embedding', 'encoding', 'engineering', 'entry', 'error', 'evaluation', 'experiment', 'expertise', 'file',
          'filter', 'form', 'frame', 'framework', 'frontend', 'function', 'generation', 'generator', 'identification',
          'implementation', 'improvement', 'inference', 'information', 'input', 'kernel', 'knowledge', 'label', 'layer',
          'learning', 'length', 'likelihood', 'location', 'mapping', 'method', 'model', 'module', 'naturalness',
          'network', 'nonlinearity', 'optimization', 'output', 'pair', 'parameter', 'pipeline', 'posterior',
          'prediction', 'process', 'processing', 'quality', 'realization', 'recognition', 'representation', 'research',
          'result', 'sample', 'score', 'segment', 'sequence', 'set', 'setting', 'signal', 'step', 'string', 'study',
          'symbol', 'synthesis', 'synthesizer', 'system', 'tagger', 'target', 'task', 'technique', 'technique',
          'technology', 'test', 'tilt', 'token', 'tool', 'toolkit', 'track', 'training', 'transcription', 'transfer',
          'transform', 'translation', 'unit', 'value', 'variation']


def get_files_from_directory(path):
    """Get all files from directory"""
    return [f for f in listdir(path) if isfile(join(path, f))]


# Read data from lexicon
def read_data(file):
    """Read data file with pandas dataframe"""
    return pd.read_csv(file, sep='\t', encoding='unicode_escape')


def select_data(dataframe):
    """Lemmatization of lexicon with scapy"""
    terms = dataframe['pilot']
    lemma = []
    for el in terms:
        doc = spacy_nlp(el.lower())
        tmp = [token.lemma_ for token in doc]
        lemma = [l.replace(' - ', '-') for l in lemma]
        lemma.append(' '.join(tmp))
    df = pd.DataFrame({'pattern': dataframe['pattern'], 'pilot': dataframe['pilot'], 'lemma': lemma})
    return df


# Extract text
def read_file(file):
    with open(file, 'r') as f:
        return f.read()


def lemma_posttag(file):
    """Convert post-tag scapy into corresponding pattern from lexicon"""
    text = read_file(file)
    doc_a = spacy_nlp(text)
    doc = spacy_nlp(text.lower())
    new_pos = []
    pos = []
    lemma = []
    t = []
    original = [token.text for token in doc_a]
    for token in doc:
        t.append(token.text)
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            new_pos.append('N')
        elif token.pos_ == 'VERB':
            new_pos.append('V')
        elif token.pos_ == 'ADJ':
            new_pos.append('A')
        elif token.pos_ == 'CCONJ' or token.pos_ == 'SCONJ':
            new_pos.append('C')
        elif token.pos_ == 'PART' or token.pos_ == 'ADP':
            new_pos.append('P')
        else:
            new_pos.append('')
    # print(len(original))
    # print(len(lemma))
    # print(len(t))
    # print(len(pos))
    # print(len(new_pos))
    frame = pd.DataFrame({'tokens': original, 'tokens_lower': t, 'lemma': lemma, 'pos': pos, 'pattern': new_pos})
    return frame


def rules(terms_dataframe, text_dataframe):
    """Define rules from terms according to their pattern"""
    new_terms = []
    for terms in terms_dataframe['lemma']:
        # Get the same structure of terms as in text dataframe
        tmp = ' '.join(terms.split('-'))
        new_terms.append(tmp.split(' '))
    for i, token in enumerate(text_dataframe['lemma']):
        for j, t in enumerate(new_terms):
            # Case 1: term of size 3 seperated by dashes (ex: text-to-speech) and followed by 1, 2 Nouns or 1 Adj and 1 Noun is a term 
            if len(t) == 3 and len(text_dataframe['lemma']) >= i + 5:
                if token == t[0] and text_dataframe['lemma'][i + 1] == '-' and (
                        text_dataframe['lemma'][i + 2] == 'to' or text_dataframe['lemma'][i + 2] == 'of' or
                        text_dataframe['lemma'][i + 2] == 'by' or text_dataframe['pattern'][i + 2] == 'N') and \
                        text_dataframe['lemma'][i + 3] == '-' and text_dataframe['lemma'][i + 4] == t[2]:
                    # followed by 2 nouns (ex: text-to-speech modal synthesis)
                    if (text_dataframe['pattern'][i + 5] == 'N' or text_dataframe['pattern'][i + 4] == 'A') and \
                            text_dataframe['pattern'][i + 6] == 'N':
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 6] = text_dataframe['tokens'][i + 6] + ']'
                    elif text_dataframe['pattern'][i + 5] == 'N' or text_dataframe['pattern'][i + 5] == 'A':
                        # followed by 1 noun (ex: text-to-speech system)
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 5] = text_dataframe['tokens'][i + 5] + ']'
                    else:
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 4] = text_dataframe['tokens'][i + 4] + ']'
            # Case 2: term of size 2 separated by dashes (ex: encoder-decoder) and followed by 0,1,2 or 3 nouns is a term
            if len(t) >= 2 and len(text_dataframe['lemma']) >= i + 3 and i != 0:
                if token == 'front' and text_dataframe['lemma'][i + 1] == '-' and text_dataframe['lemma'][
                    i + 2] == 'end':
                    if text_dataframe['pattern'][i - 1] == 'N':
                        text_dataframe['tokens'][i - 1] = '[' + text_dataframe['tokens'][i - 1]
                        text_dataframe['tokens'][i + 2] = text_dataframe['tokens'][i + 2] + ']'
                if token == t[0] and text_dataframe['lemma'][i + 1] == '-' and text_dataframe['lemma'][i + 2] == t[1]:
                    # followed by 3 nouns (ex: HMM-based generation synthesis approach)
                    if text_dataframe['pattern'][i + 3] == 'N' and text_dataframe['pattern'][i + 4] == 'N' and \
                            text_dataframe['pattern'][i + 5] == 'N':
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 5] = text_dataframe['tokens'][i + 5] + ']'
                    # followed by 2 nouns (ex: HMM-based generation synthesis)
                    elif (text_dataframe['pattern'][i + 3] == 'N' or text_dataframe['pattern'][i + 3] == 'A' or
                          text_dataframe['pattern'][i + 3] == 'V') and text_dataframe['pattern'][i + 4] == 'N':
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 4] = text_dataframe['tokens'][i + 4] + ']'
                    # followed by 1 noun (ex: cross-lingual adaptation)
                    elif text_dataframe['pattern'][i + 3] == 'N':
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 3] = text_dataframe['tokens'][i + 3] + ']'
                    # followed by nothing (ex: mel-spectrogram)
                    else:
                        text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                        text_dataframe['tokens'][i + 2] = text_dataframe['tokens'][i + 2] + ']'
        if (
                token == 'data' or token == 'voice' or token == 'datum' or token == 'speaker' or token == 'dataset' or token == 'database' or token == 'feature' or token == 'corpus') and i != 0 and len(
            text_dataframe['lemma']) >= i + 1:
            if text_dataframe['pattern'][i - 1] == 'N' or text_dataframe['pattern'][i - 1] == 'A':
                text_dataframe['tokens'][i - 1] = '[' + text_dataframe['tokens'][i - 1]
                text_dataframe['tokens'][i] = text_dataframe['tokens'][i] + ']'
            elif text_dataframe['pattern'][i + 1] == 'N':
                text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                text_dataframe['tokens'][i + 1] = text_dataframe['tokens'][i + 1] + ']'
        if i != 0:
            if text_dataframe['lemma'][i - 1] in rule_adj and '[' in text_dataframe['tokens'][i]:
                text_dataframe['tokens'][i - 1] = '[' + text_dataframe['tokens'][i - 1] + ']'
            elif i >= 3 and text_dataframe['lemma'][i - 1] in rule_adj and text_dataframe['lemma'][
                i - 3] == 'non' and '[' in text_dataframe['tokens'][i]:
                text_dataframe['tokens'][i - 3] = '[' + text_dataframe['tokens'][i - 3]
                text_dataframe['tokens'][i - 3] = text_dataframe['tokens'][i - 1] + ']'


def annotate(terms_dataframe, text_dataframe):
    """Annotate the terms of the text thanks to list of terms + applied rules"""
    rules(terms_dataframe, text_dataframe)  # apply rules
    for i, token in enumerate(text_dataframe['lemma']):
        for term in terms_dataframe['lemma']:
            term = term.split(' ')
            # Case 1: if terms of length 4, we check if each word from text corresponds to each word in the term
            if len(term) == 4:
                term_1 = term[0]
                if token == term_1 and len(text_dataframe['lemma']) >= i + 4:
                    if text_dataframe['lemma'][i + 1] == term[1] and text_dataframe['lemma'][i + 2] == term[2] and \
                            text_dataframe['lemma'][i + 3] == term[3]:
                        if text_dataframe['lemma'][i + 4] in rule_4:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 4] = text_dataframe['tokens'][i + 4] + ']'
                        else:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 3] = text_dataframe['tokens'][i + 3] + ']'
            # Case 2: terms of length 3
            elif len(term) == 3:
                term_1 = term[0]
                if token == term_1 and len(text_dataframe['lemma']) > i + 3:
                    if text_dataframe['lemma'][i + 1] == term[1] and text_dataframe['lemma'][i + 2] == term[2]:
                        if text_dataframe['lemma'][i + 3] in rule_4:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 3] = text_dataframe['tokens'][i + 3] + ']'
                        else:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 2] = text_dataframe['tokens'][i + 2] + ']'
            # Case 3: terms of length 2
            elif len(term) == 2:
                if token == term[0] and len(text_dataframe['lemma']) > i + 2:
                    if text_dataframe['lemma'][i + 1] == term[1]:
                        if text_dataframe['lemma'][i + 2] in rule_4:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 2] = text_dataframe['tokens'][i + 2] + ']'
                        else:
                            text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                            text_dataframe['tokens'][i + 1] = text_dataframe['tokens'][i + 1] + ']'
            # Case 4: term of length 1
            elif token == term[0] and i > 1 and text_dataframe['lemma'][i - 1] == 'of' and text_dataframe['lemma'][
                i - 2] == 'sequence':
                text_dataframe['tokens'][i - 2] = '[' + text_dataframe['tokens'][i - 2]
                text_dataframe['tokens'][i] = text_dataframe['tokens'][i] + ']'
            elif token == term[0] and len(term) == 1 and len(text_dataframe['lemma']) >= i + 2 and \
                    text_dataframe['lemma'][i + 1] == ')':
                if text_dataframe['lemma'][i + 2] in rule_4:
                    text_dataframe['tokens'][i - 1] = '[' + text_dataframe['tokens'][i - 1]
                    text_dataframe['tokens'][i + 2] = text_dataframe['tokens'][i + 2] + ']'
                else:
                    text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i] + ']'
            elif token == term[0] and len(term) == 1 and len(text_dataframe['lemma']) >= i + 1:
                if text_dataframe['lemma'][i + 1] in rule_4:
                    text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i]
                    text_dataframe['tokens'][i + 1] = text_dataframe['tokens'][i + 1] + ']'
                else:
                    text_dataframe['tokens'][i] = '[' + text_dataframe['tokens'][i] + ']'
        if i != 0:
            if text_dataframe['lemma'][i - 1] in rule_adj and '[' in text_dataframe['tokens'][i]:
                text_dataframe['tokens'][i - 1] = '[' + text_dataframe['tokens'][i - 1] + ']'
            elif i >= 3 and text_dataframe['lemma'][i - 1] in rule_adj and text_dataframe['lemma'][
                i - 3] == 'non' and '[' in text_dataframe['tokens'][i]:
                text_dataframe['tokens'][i - 3] = '[' + text_dataframe['tokens'][i - 3]
                text_dataframe['tokens'][i - 3] = text_dataframe['tokens'][i - 1] + ']'
    return text_dataframe


def construct_annotated_text(text_dataframe):
    """Return the text from the annotated text dataframe with the correct annotation of brackets"""
    content = ' '.join(text_dataframe['tokens'].to_list())
    compt = 0
    compt2 = 0
    string = ''
    for i in content:
        if i == '[':
            if compt == 0:
                compt += 1
                string += i
            elif compt >= 1:
                compt += 1
        elif i == ']':
            if compt - 1 != compt2:
                compt2 += 1
            else:
                string += i
                compt = 0
                compt2 = 0
        else:
            string += i
    string = string.replace('] [', ' ')
    string = string.replace(' .', '.')
    string = string.replace(' ’', '’')
    string = string.replace(' ,', ',')
    string = string.replace(' - ', '-')
    string = string.replace('( ', '(')
    string = string.replace(' )', ')')
    string = string.replace(']-[', '-')
    string = string.replace('.]', '].')
    string = string.replace('\n ', '\n')
    return string


def tagging_IOB(string):
    """Tagging the terms into IOB"""
    string = string.replace('\n', '\n ')
    is_term = False
    string_tag = string.split(' ')
    annotated = []
    for k, l in enumerate(string_tag):
        if '[' in l and ']' in l:
            for i, j in enumerate(l):
                if l[i] == ']':
                    annotated.append(l[:i] + ' (B)]' + l[i + 1:])
        else:
            if '[' in l and is_term is False:
                annotated.append(l + ' (B)')
                is_term = True
            elif is_term and ']' not in l:
                annotated.append(l + ' (I)')
            elif is_term and ']' in l:
                for m, n in enumerate(l):
                    if l[m] == ']':
                        annotated.append(l[:m] + ' (I)]' + l[m + 1:])
                is_term = False
            else:
                if "\n" not in l and l != 0 and l not in s.punctuation:
                    annotated.append(l + ' (O)')
                else:
                    annotated.append(l)
    iob_string = ' '.join(annotated)
    iob_string = iob_string.replace('\n ', '\n')
    return iob_string


def POS_tags(string):
    """Tagging the terms into IOB and POS"""
    tagged = []
    doc = spacy_nlp(string)
    for el in doc:
        if el.text not in ['B', 'O', 'I', '\n'] and el.text not in s.punctuation and len(el.text) != 0:
            tagged.append(el.text + ' ' + el.pos_)
        else:
            tagged.append(el.text)
    all_tags = ' '.join(tagged)
    all_tags = all_tags.replace('( ', '(')
    all_tags = all_tags.replace(' )', ')')
    all_tags = all_tags.replace('[ ', '[')
    all_tags = all_tags.replace(' ]', ']')
    all_tags = all_tags.replace('SPACE', '')
    all_tags = all_tags.replace('\n  ', '\n')
    return all_tags


def annotate_terms(text_file, output_file):
    """Annotate the terms of a text and save it in a file"""
    init_data = read_data('lexicon.tsv')
    data = select_data(init_data)
    text_dataframe = lemma_posttag(text_file)
    annotate(data, text_dataframe)
    annotation = construct_annotated_text(text_dataframe)
    with open(output_file, 'w') as f:
        f.write(annotation)
        print("Your file has been annotated.")


def annotate_iob(text_file, output_file):
    """Annotate the terms + add IOB tag of a text and save it in a file"""
    init_data = read_data('lexicon.tsv')
    data = select_data(init_data)
    text_dataframe = lemma_posttag(text_file)
    annotate(data, text_dataframe)
    annotation = construct_annotated_text(text_dataframe)
    iob_text = tagging_IOB(annotation)
    with open(output_file, 'w') as f:
        f.write(iob_text)
        print("Your file has been annotated and IOB_tagged.")


def annotate_iob_pos(text_file, output_file):
    """Annotate the terms + add IOB tag + POS tag of a text and save it in a file"""
    init_data = read_data('lexicon.tsv')
    data = select_data(init_data)
    text_dataframe = lemma_posttag(text_file)
    annotate(data, text_dataframe)
    annotation = construct_annotated_text(text_dataframe)
    iob_text = tagging_IOB(annotation)
    pos_text = POS_tags(iob_text)
    with open(output_file, 'w') as f:
        f.write(pos_text)
        print("Your file has been annotated, IOB_tagged and POS_tagged.")


if __name__ == "__main__":
    # path_1 = '/home/macaire/Bureau/M2_NLP/Terminology/terminology_project/Corpus/train/'
    # path_2 = '/home/macaire/Bureau/M2_NLP/Terminology/terminology_project/Corpus/annot_IOB_POS/'
    # files = get_files_from_directory('/home/macaire/Bureau/M2_NLP/Terminology/terminology_project/Corpus/train/')
    # for i in files:
    #     try:
    #         annotate_iob_pos(path_1+i, path_2+i)
    #     except:
    #         print(i)
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--annotation_type', type=int,
                        help='choose the type of annotation: 1=terms, 2=IOB, 3=POS')
    parser.add_argument('-i', '--input_text', type=str, help='directory to load text')
    parser.add_argument('-o', '--output_text', type=str, help='directory to save text')

    args = parser.parse_args()
    if len(sys.argv) <= 1:
        print("You should run this script like: python termsprocessing.py -n num_type  -i input.txt -o output.txt")
        exit(1)
    if args.annotation_type == 1:
        annotate_terms(args.input_text, args.output_text)
    elif args.annotation_type == 2:
        annotate_iob(args.input_text, args.output_text)
    elif args.annotation_type == 3:
        annotate_iob_pos(args.input_text, args.output_text)
    else:
        annotate_terms(args.input_text, args.output_text)
