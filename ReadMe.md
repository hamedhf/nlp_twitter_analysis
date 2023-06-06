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
```

```bash
python main.py break-by-word path-to-cleaned-csv
```

```bash
python main.py break-by-sentence path-to-punc-csv
```

```bash
python main.py get-stats file-timestamp
```

```bash
python main.py generate-pdf-report file-timestamp
```

## Hugging Face Dataset

[Dataset Link](https://huggingface.co/datasets/hamedhf/nlp_twitter_analysis/tree/main)

## License

[MIT](https://choosealicense.com/licenses/mit/)