import openai
import backoff
from openai.error import RateLimitError
from glob import glob
import json
import os
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
import argparse
import os
import nltk
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()



openai.api_key = os.environ["OPENAI_API_KEY"]

@backoff.on_exception(backoff.expo, RateLimitError)
def response_API(prompt, myKwargs = {}):
    
    try:
        response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=prompt
        )   
        
    except openai.error.APIConnectionError:
        print("Failed")
        return None
    except openai.error.InvalidRequestError as e:
        print("InvalidRequestError Failed")
        print(e)
        return None
    
    return response['choices'][0]['message']['content']



def format_prompt_summary(articles, max_tokens=10000):
    # messages=[{"role": "system", "content": "You are a journalist in a news room. A new event has occurred, you are given news articles from the competitors and are given the following task."}]
    

    sentences_string = ""    
    for article_idx, article in enumerate(articles):
        sentences_string += f"Article {article_idx} \n ==== \n"
        
        if len(word_tokenize(article)) > max_tokens:
            article_string = ' '.join(word_tokenize(article)[:max_tokens])
        else:
            article_string = article
        sentences_string += article_string
        sentences_string += "\n"
    
    
    messages= [
        {"role": "system", "content": "You are a helpful assistant that reads instructions carefully."},
        {"role": "user", "content": f"""Read the following news articles. Produce a summary that only covers the diverse and conflicting information across the following articles, without discussing the information all articles agree upon. Elaborate when you summarize diverse or conflicing information by stating what information different sources cover and how is the information diverse or conflicting. You must give your in a structured format: ```Summary: [your summary]```, where [your summary] is your generated summary.
                                                    ==========
                                                    {sentences_string}
                                                    ==========
                                                    Remember, your output should be a summary that discusses and elaborates the diverse and conflicting information presented across the articles. You need to elaborate on the differences rather than only mentioning which topic they differ. Don't worry about the summary being too lengthy.
                                                    """}]                                                
    
    return messages

with open("../data/diverse_summ.json", "r") as f:
    diverse_summ = json.load(f)


for instance in tqdm(diverse_summ, desc='Generating summaries: '):
    
    
    # Gather eid and articles
    eid = instance['eid']

    # We only take the content for each article.
    articles = [article['content'] for article in instance["articles"]]

    prompt = format_prompt_summary(articles)
    try:
        generated_summary = response_API(prompt)
    except:
        generated_summary = None

    if generated_summary is None:

        # Truncate the input until it fits the context window.
        for max_tokens in range(16000, 1, -1):
            prompt = format_prompt_summary(articles, max_tokens)
            try:
                generated_summary = response_API(prompt)
            except:
                generated_summary = None


    with open(args.output_path,'a') as f:
        f.write(json.dumps({'summary': generated_summary,
                            'eid':eid}) + '\n')

            
            
