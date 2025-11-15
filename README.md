# Licence plate object detection

## Table of Contents
- [Description](#description)
- [Data source](#data-source)
- [How to run the code](#how-to-run-the-code)
- [Author](#author)

## Description
This is one of my project in university. The goal of this project is to make a regression deep learning model that can predict the bounding box of license plates on cars.

![](book_recommender_demo.png)

## Data source
The data a public dataset on hugging face which you can find in this link </br>
[Kaggle 7K Books Datasets](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

## How to run the code:
1. Download Ollama at: [https://ollama.com/](https://ollama.com/)
2. In the CLI: </br>
`ollama pull llama3.2` </br>
`python -m venv venv_name` </br>
`venv_name\Scripts\activate` </br>
`pip install -r requirements.txt` </br>
3. Run the notebooks consecutively in this order:
- data-exploration.ipynb
- vector-search.ipynb
- text-classification.ipynb
- sentiment-analysis.ipynb
4. Run the app </br>
`streamlit run app.py`


##  Author
- Chu Cao Nguyen - nguyenmilan203@gmail.com
  
