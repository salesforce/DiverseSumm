import openai
import backoff
from openai.error import RateLimitError
import json
import os
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
import argparse
import nltk
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt')

# Set the OpenAI API key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# Parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--generated_question_path', type=str, required=True, help="The file produced by gpt_qg.py.")
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

# Exponential backoff for handling Rate Limit errors from OpenAI API
@backoff.on_exception(backoff.expo, RateLimitError)
def response_API(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=prompt
        )   
        return response['choices'][0]['message']['content']
    except (openai.error.APIConnectionError, openai.error.InvalidRequestError) as e:
        print(f"API call failed: {e}")
        return None

# Function to format the prompt for the OpenAI API
def format_prompt(article, question, max_token=6000):
    if len(word_tokenize(article)) > max_token:
        article = ' '.join(word_tokenize(article)[:max_token])
        
    messages = [
        {"role": "user", "content": (
            f"Read the following news article and answer only the question '{question}'. "
            "Extract the exact sentence from the article changing up to 5 words. You should include ALL the answers that can be found in the article and must give your answers in a structured format: "
            "'Answer 1: [extracted answer 1] \n Answer 2: [extracted answer 2] ...'. If the article contains no information to the given question, write: 'No Answer'."
            "==========="
            f"{article}"
        )}
    ]
    return messages

# Function to load data from JSON or JSONL files
def load_json(filepath):
    with open(filepath, "r") as f:
        if filepath.endswith('jsonl'):
            return [json.loads(line) for line in f.readlines()]
        else:
            return json.load(f)

def main():
    questions_with_events = load_json(args.generated_question_path)
    articles = load_json("PATH/TO/ARTICLES.json")
    aid2article = {a["_id"]: a for a in articles}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for event in tqdm(questions_with_events):
            eid = event['eid']
            aids = event['aids']
            questions = event['questions']
            articles_content = [aid2article[aid]['content'] for aid in aids]

            for question in questions:
                futures.append(executor.submit(process_question, eid, aids, question, articles_content, args.output_path))
        
        for future in futures:
            future.result()

def process_question(eid, aids, question, articles, output_path):
    all_answers = []
    for article in articles:
        prompt = format_prompt(article=article, question=question)
        answers = response_API_with_retry(prompt, article, question)
        all_answers.append(answers)

    answers_for_this_question = {
        'eid': eid,
        'aids': aids,
        'question': question,
        'answers': all_answers
    }

    with open(output_path, 'a') as f:
        f.write(json.dumps(answers_for_this_question) + '\n')

def response_API_with_retry(prompt, article, question):
    answers = response_API(prompt)
    if answers is None:
        for max_token in range(16000, 1, -1):
            prompt = format_prompt(article=article, question=question, max_token=max_token)
            answers = response_API(prompt)
            if answers is not None:
                break
    return answers

if __name__ == "__main__":
    main()