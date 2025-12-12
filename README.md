# Retrieval-Augmented Generation

Custom RAG system for answering rules question in boardgames.

## Example

Application to the boardgame [Hegemony (2023)](https://boardgamegeek.com/image/5715770/hegemony-lead-your-class-to-victory), a complex heavy asymetric game with a 40-page [rulebook](https://hegemonicproject.com/wp-content/uploads/2023/04/Hegemony-English-Rulebook-v1.2.pdf).

```sh
⏳ populate_vector_store: 0.24s
⏳ single_query_retrival: 0.24s
⏳ expand_query: 9.70s
⏳ multi_query_retrieval: 0.03s
⏳ rerank_chunks: 0.03s
⏳ get_answer_from_query: 28.33s
```
- **Query:**  how does the working class score victory points (VPs)?
- **Answer:**  The Working Class scores victory points (VPs) through increasing their Prosperity by providing Health, Education, or Luxury to its people, establishing Trade Unions in Industries where many of its Workers are employed, having Socialist Policies in the Politics Table (section A), and having money remaining on their board at the end of the game (1 VP for every 10 remaining, up to 15 VP).

![hegemony](https://github.com/user-attachments/assets/18b46c8b-f0e7-4090-9256-721579df1e49)

## Python environment

```sh
python -m pip install --upgrade setuptools pip
mkdir .venv
pipenv install -d --python 3.12
```

## Running the main script

```sh
pipenv shell
python main.py
```
