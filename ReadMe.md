# nlp_twitter_analysis

This is an NLP (Natural Language Processing) project that focuses on gathering data from Twitter and labeling the topics
of
the tweets using ChatGPT with supervision of a human annotator.

## Installation

 ```bash
poetry install
 ``` 

```bash
poetry shell
```

```bash
cd src
copy .env.sample .env
touch users.csv
```

## How it works

**Crawl -> Label -> Clean/Punc -> Break by word/sentence -> Analyze**

## Usage

```bash
python main.py --help
```

```bash
python main.py scrape-twitter
```

```bash
python main.py example-labeling
```

```bash
python main.py label-data
```

```bash
python main.py clean-data path-to-labeled-csv

example: python src/main.py clean-data ../data/raw/labeled_2023-06-02-10-27-57.csv
```

```bash
python main.py break-by-word path-to-cleaned-csv

example: python src/main.py break-by-word ../data/clean/cleaned_2023-06-02-10-27-57.csv
```

```bash
python main.py break-by-sentence path-to-punc-csv

example: python src/main.py break-by-sentence ../data/clean/punc_2023-06-02-10-27-57.csv
```

```bash
python main.py get-stats file-timestamp

example: python src/main.py get-stats 2023-06-02-10-27-57
```

```bash
python main.py generate-pdf-report file-timestamp

example: python src/main.py generate-pdf-report 2023-06-02-10-27-57
```

## Hugging Face Dataset

[Dataset Link](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

## License

[MIT](https://choosealicense.com/licenses/mit/)