from typing import *
from tqdm import tqdm
from striprtf.striprtf import rtf_to_text
import spacy_udpipe
import subprocess
import os

LANGUAGE_CODES = {
    'Austria'       : "de",  # assuming german,
    'Belgium'       : 'fr',  # assuming french
    'Croatia'       : "hr",
    'Czechia'       : "cs",
    'Estonia'       : "et",
    'Finland'       : "fi",
    'France'        : "fr",
    'Denmark'       : "da",
    'Greece'        : "grc",
    'Germany'       : "de",
    'Hungary'       : "hu",
    'Ireland'       : "ga",
    'Italy'         : "it",
    'Latvia'        : "lv",
    'Lithuania'     : "lt",
    'Netherlands'   : "nl",
    'Poland'        : "pl",
    'Portugal'      : "pt",
    'Romania'       : "ro",
    'Slovakia'      : "sk",
    'Slovenia'      : "sl",
    'Spain'         : "es",
    'Sweden'        : "sv",
    'UK'            : "en"
}

ROOT_FOLDER = './'


def convert_rtf(lang: str) -> List[str]:
    lang_dir = os.path.join(ROOT_FOLDER, lang)
    txt_files = [f for f in os.listdir(lang_dir) if f.endswith('txt')]
    

def get_txt_paths_from_lang(path: str, ext: str) -> List[str]:
    txt_files = [f for f in os.listdir(path) if f.endswith(ext)]
    return [os.path.join(path, f) for f in txt_files]


def do_a_lang(lang: str, path: str, from_pdf: bool = False) -> Dict[str, Dict[str, List[List[str]]]]:
    code = LANGUAGE_CODES[lang]
    spacy_udpipe.download(code) # download appropriate model
    nlp = spacy_udpipe.load(code, ignore_tag_map=True)

    txt_file_paths = get_txt_paths_from_lang(path, ext = 'pdf' if from_pdf else 'txt')

    named_dict = {}
    for path in tqdm(txt_file_paths):
        file_name = path.split('/')[-1]
        
        if not from_pdf:
            with open(path, 'r+') as f:
                lines = [l for l in (line.strip() for line in f) if l]
        else:
            try:
                lines = pdfparser(path)
            except:
                continue
                
        # apply Spacy - UDPipe
        lines = [nlp(line[:1000000]) for line in lines]
        
        # get all elements
        txt = [[token.text for token in line] for line in lines] 
        lemmas = [[token.lemma_ for token in line] for line in lines]
        poss = [[token.pos_ for token in line] for line in lines]
        deps = [[token.dep_ for token in line] for line in lines]

        # save elements per text file
        named_dict[file_name] = {'text': txt, 'lemma': lemmas, 'pos': poss, 'dep': deps}

    return named_dict


def write_processed(proc: Dict[str, Any], write_path: str):
    for filename, stuff in proc.items():
        filename = filename.split('.')[0]
        texts, lemmas, poss, deps = stuff.values()
        _path = os.path.join(write_path, filename)
        for t, l, p, d in zip(texts, lemmas, poss, deps):
                _txt = ' '.join(t)
                _lemma = ' '.join(l)
                _pos = ' '.join(p)
                _dep = ' '.join(d)
                _write = '\t'.join([_txt, _lemma, _pos, _dep])
                with open(_path + '.txt', 'a+') as f:
                    f.write(_write)
                    f.write('\n')


def docx_to_txt(docx_path: str, txt_path: str):
    from pypandoc import convert_file
    docxs = [f for f in os.listdir(docx_path) if f.endswith('docx')]
    file_paths = [os.path.join(docx_path, f) for f in docxs]
    out_paths = [os.path.join(txt_path, f.split('.')[0] + '.txt') for f in docxs]
    for fp, op in zip(file_paths, out_paths):
        out = convert_file(fp, 'plain', outputfile=op)
        assert out == ""


def pdfparser(data: str) -> List[str]:
    import sys
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfpage import PDFPage
    from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
    from pdfminer.layout import LAParams
    import io
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()

    return [l for l in (line.strip() for line in data.split('\n')) if l]