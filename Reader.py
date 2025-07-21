import os

from docx import Document
# from hazm import SentenceTokenizer
import nltk
import Similarity
import fitz
import re
import stanza

nlp_rus = stanza.Pipeline(lang='ru')
nlp_per = stanza.Pipeline(lang='fa')

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()
    # full_text = full_text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("(", " ").replace(")", " ")
    # full_text_with_one_space =  re.sub(r'\s+', ' ', full_text)
    #
    # # something like 2. 4.
    # full_text_with_no_period_number = re.sub(r'\d+\.', '', full_text_with_one_space)
    return full_text

def extract_tok_sent(file_path):
    sentences = []
    with open(file_path, encoding="utf8") as f:
        sentences = f.readlines()
        f.close()
    return sentences


def clean_text(text):
    # Replace unwanted characters
    pattern = r'[()\-:;,\d.!?\'"@#%&*[\]{}<>/_]'
    cleaned_text = re.sub(pattern, '', text)
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def split_persian_sentences(text):
    # # tokenizer = SentenceTokenizer()
    # # sentences = tokenizer.tokenize(text)
    # # return sentences
    # # pattern = r'[;.\(\)]'
    # pattern = r'[.]'
    # split_text = re.split(pattern, text)
    #
    # # Remove any empty strings resulting from consecutive delimiters
    # split_text = [segment.strip() for segment in split_text if segment.strip()]
    # split_text = [clean_text(text) for text in split_text]
    # return split_text
    doc_farsi = nlp_per(text)
    sentences = []
    for sentence in doc_farsi.sentences:
        sentences.append(sentence.text)
    sentences = [segment.replace("\n", " ") for segment in sentences if segment.strip()]
    return sentences


def split_russian_sentences(text):
    # # sentences = sent_tokenize(text, language='russian')
    # # return sentences
    # # pattern = r'[;.\(\)]'
    # pattern = r'[.]'
    # split_text = re.split(pattern, text)
    #
    # # Remove any empty strings resulting from consecutive delimiters
    # split_text = [segment.strip() for segment in split_text if segment.strip()]
    # split_text = [clean_text(text) for text in split_text]
    # return split_text
    doc_russian = nlp_rus(text)
    sentences = []
    for sentence in doc_russian.sentences:
        sentences.append(sentence.text)
    sentences = [segment.replace("\n", " ") for segment in sentences if segment.strip()]
    return sentences


def walk_on_directory_and_check_similarity(base_directory):
    full_matches = []
    count =0
    for root, dirs, files in os.walk(base_directory):
        for dir in dirs:
            for inner_root, inner_dirs, inner_files in os.walk(os.path.join(root, dir)):
                if (len(inner_files)!=2):
                    continue
                persian_path = os.path.join(inner_root, inner_files[0])
                russian_path = os.path.join(inner_root, inner_files[1])
                print(persian_path)

                if ".docx" in russian_path:
                    russian_text = extract_text_from_docx(russian_path)
                else:
                    russian_text = extract_text_from_pdf(russian_path)
                if ".docx" in persian_path:
                    persian_text = extract_text_from_docx(persian_path)
                else:
                    persian_text = extract_text_from_pdf(persian_path)


                persian_sentences = split_persian_sentences(persian_text)
                russian_sentences = split_russian_sentences(russian_text)
                # persian_path = os.path.join(inner_root, "f.sent.txt")
                # russian_path = os.path.join(inner_root, "r.sent.txt")
                # persian_sentences = extract_tok_sent(persian_path)
                # russian_sentences = extract_tok_sent(russian_path)
                persian_sentences1 = [clean_text(sentence) for sentence in persian_sentences]
                russian_sentences1 = [clean_text(sentence) for sentence in russian_sentences]
                russian_embeddings, persian_embeddings = Similarity.get_embeddings(russian_sentences1, persian_sentences1)
                print (count)
                count=count+1
                cosine_scores = Similarity.get_similarity(russian_embeddings, persian_embeddings)
                matches = []
                for i in range(len(russian_sentences1)):
                    score = cosine_scores[i].max().item()
                    best_match_idx = cosine_scores[i].argmax().item()
                    matches.append((russian_sentences1[i], persian_sentences1[best_match_idx], score, russian_path))
                full_matches.extend(matches)
    return full_matches

def check_similarity(source_path, target_path):
    if ".docx" in source_path:
        russian_text = extract_text_from_docx(source_path)
    else:
        russian_text = extract_text_from_pdf(source_path)
    if ".docx" in target_path:
        persian_text = extract_text_from_docx(target_path)
    else:
        persian_text = extract_text_from_pdf(target_path)


    persian_sentences = split_persian_sentences(persian_text)
    russian_sentences = split_russian_sentences(russian_text)
    # persian_path = os.path.join(inner_root, "f.sent.txt")
    # russian_path = os.path.join(inner_root, "r.sent.txt")
    # persian_sentences = extract_tok_sent(persian_path)
    # russian_sentences = extract_tok_sent(russian_path)
    persian_sentences1 = [clean_text(sentence) for sentence in persian_sentences]
    russian_sentences1 = [clean_text(sentence) for sentence in russian_sentences]
    russian_embeddings, persian_embeddings = Similarity.get_embeddings(russian_sentences1, persian_sentences1)
    cosine_scores = Similarity.get_similarity(russian_embeddings, persian_embeddings)
    matches = []
    for i in range(len(russian_sentences1)):
        score = cosine_scores[i].max().item()
        best_match_idx = cosine_scores[i].argmax().item()
        matches.append((russian_sentences1[i], persian_sentences1[best_match_idx], score, source_path))
    return matches

# def extract_text_from_file(pdf_path):
#     doc = fitz.open(pdf_path)
#     full_text = ""
#
#     for page in doc:
#         full_text += page.get_text()
#     full_text = full_text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("(", " ").replace(")", " ")
#     full_text_with_one_space =  re.sub(r'\s+', ' ', full_text)
#
#     # something like 2. 4.
#     full_text_with_no_period_number = re.sub(r'\d+\.', '', full_text_with_one_space)
#     return full_text_with_no_period_number
#
#
# def split_text_to_paragraphs(full_text):
#     paragraphs = full_text.split(".")
#     return paragraphs
