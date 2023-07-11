import os
import re


def check_necessary_files():
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if not os.path.exists('../data/raw'):
        os.mkdir('../data/raw')

    if not os.path.exists('../data/clean'):
        os.mkdir('../data/clean')

    if not os.path.exists('../data/augment'):
        os.mkdir('../data/augment')

    if not os.path.exists('../data/wordbroken'):
        os.mkdir('../data/wordbroken')

    if not os.path.exists('../data/sentencebroken'):
        os.mkdir('../data/sentencebroken')

    if not os.path.exists('../data/languagemodel'):
        os.mkdir('../data/languagemodel')

    if not os.path.exists('../data/split'):
        os.mkdir('../data/split')

    if not os.path.exists('../stats'):
        os.mkdir('../stats')

    if not os.path.exists('../models'):
        os.mkdir('../models')

    if not os.path.exists('../models/word2vec'):
        os.mkdir('../models/word2vec')

    if not os.path.exists('../models/parsbert'):
        os.mkdir('../models/parsbert')

    if not os.path.exists('../models/gpt2'):
        os.mkdir('../models/gpt2')

    if not os.path.exists('../models/gpt2/base_persian'):
        os.mkdir('../models/gpt2/base_persian')

    if not os.path.exists('../models/huggingface_cache'):
        os.mkdir('../models/huggingface_cache')

    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    if not os.path.exists('./users.csv'):
        raise FileNotFoundError("users.csv not found. Please create it.")

    if not os.path.exists('./.env'):
        raise FileNotFoundError(".env not found. Please create it like .env.example.")


def latex_pdf_report(phase: int, file_timestamp: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    latex_source_path = f'{root_dir}/src/latex/report.tex' if phase == 1 else f'{root_dir}/src/latex/report_final.tex'

    with open(latex_source_path, 'r') as latex_file:
        latex_source = latex_file.read()

    # replacing variables \def\timestamp{2023-06-02-10-27-57}
    latex_source = latex_source.replace(
        r'\def\timestamp{2023-06-02-10-27-57}',
        r'\def\timestamp{' + file_timestamp + '}'
    )

    tmp_file_path = f'{root_dir}/src/latex/tmp.tex'
    with open(tmp_file_path, 'w') as tmp_file:
        tmp_file.write(latex_source)

    command = f'pdflatex -output-directory={root_dir}/src/latex -jobname=Phase{phase}-Report {root_dir}/src/latex/tmp.tex'  # noqa
    os.system(command)

    pdf_save_path = root_dir + f'/Phase{phase}-Report.pdf'
    os.rename(f'{root_dir}/src/latex/Phase{phase}-Report.pdf', pdf_save_path)
    print(f"PDF report generated at {pdf_save_path}")

    pattern = r'^Phase1-Report.*' + '|' + r'^tmp.*' + '|' + r'^report.*'
    pattern += r'|' + r'^Phase2-Report.*' + '|' + r'^report_final.*'
    for f in os.listdir(f'{root_dir}/src/latex'):
        if re.search(pattern, f) and f != 'report.tex' and f != 'report_final.tex':
            os.remove(os.path.join(f'{root_dir}/src/latex', f))
