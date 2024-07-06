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
        model='gpt-4',
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



def format_prompt_extraction(article, max_tokens=4000):
    
    # Truncate the input if longer than max_tokens.    
    if len(word_tokenize(article)) > max_tokens:
        article = ' '.join(word_tokenize(article)[:max_tokens])
    
    messages= [{"role": "system", "content": "You are a helpful assistant that reads instructions carefully."},
    {"role": "user", "content": f"""Read the following news article. Extract the most important 10 sentences from the article and do not change words in the sentences. Your extracted sentence must be in a structured format: 'Sentence 1: [sentence 1] \n Sentence 2: [sentence 2] \n Sentence 3: [sentence 3] ...' where [sentence 1] should be the most important sentence.
                                                    ==========
                                                    {article}
                                                    """}]                                                
    
    return messages

def format_prompt_summary(sentences, max_sentences=10):

    # Formate input sentences from each article.
    sentences_string = ""    
    for article_idx, article_sentences in enumerate(sentences):
        sentences_string += f"Article {article_idx} \n ==== \n"
        sentences_string += ' '.join(article_sentences[:max_sentences])
        sentences_string += "\n"
    
    
    messages= [{"role": "system", "content": "You are a helpful assistant that reads instructions carefully."},
            {"role": "user", "content": f"""Read the following sentences from different articles. Produce a summary that only covers the diverse and conflicting information across the following articles, without discussing the information all articles agree upon. Elaborate when you summarize diverse or conflicing information by stating what information different sources cover and how is the information diverse or conflicting. You must give your in a structured format: ```Summary: [your summary]```, where [your summary] is your generated summary.
                                                    ==========
                                                    {sentences_string}
                                                    ==========
                                                    Remember, your output should be a summary that discusses and elaborates the diverse and conflicting information presented across the articles. You need to elaborate on the differences rather than only mentioning which topic they differ. Don't worry about the summary being too lengthy.
                                                    """}]                                                
    
    return messages

def parse_sentences(sentences_string):
    return sentences_string.split('\n')

with open("../data/diverse_summ.json", "r") as f:
    diverse_summ = json.load(f)


for instance in tqdm(diverse_summ, desc='Generating summaries: '):
    
    
    # Gather eid and articles
    eid = instance['eid']

    # We only take the content for each article
    articles = [article['content'] for article in instance["articles"]]

    # Extract important sentences from articles
    all_extracted_sentences = []        
    for article in articles:
        prompt = format_prompt_extraction(article=article)
        try:
            extracted_sentences = response_API(prompt)
        except:
            extracted_sentences = None
        if extracted_sentences is None:
            
            # Truncate the input until it fits the context window.
            for max_tokens in range(4000, 1, -1):
                
                prompt = format_prompt_extraction(article=article, max_tokens=max_tokens)
                try:
                    extracted_sentences = response_API(prompt)
                except:
                    extracted_sentences = None
                if extracted_sentences is not None:
                    break
        
        parsed_extracted_sentences = parse_sentences(extracted_sentences)

        all_extracted_sentences.append(parsed_extracted_sentences)

    # Generate summary based on articles
    prompt = format_prompt_summary(all_extracted_sentences)
    generated_summary = response_API(prompt)
    if generated_summary is None:

        # Truncate the input until it fits the context window.
        for max_sentences in range(10, 1, -1):
            prompt = format_prompt_summary(all_extracted_sentences, max_sentences)
            generated_summary = response_API(prompt)


    with open(args.output_path,'a') as f:
        f.write(json.dumps({'summary': generated_summary,
                            'extracted_sentences': all_extracted_sentences,
                            'eid':eid}) + '\n')

            
            
