import openai
import backoff
from openai.error import RateLimitError, APIConnectionError, InvalidRequestError
import json
import os
import argparse
import nltk
from nltk import word_tokenize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Download NLTK tokenizer
nltk.download('punkt')

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, required=True, help="Output file name. Should be a jsonl file.")
args = parser.parse_args()

# Set API key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Retry logic for API requests
@backoff.on_exception(backoff.expo, RateLimitError)
def response_API(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=prompt
        )
        return response['choices'][0]['message']['content']
    except (APIConnectionError, InvalidRequestError) as e:
        print(f"API Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

# Format prompt for the API
def format_prompt(articles, max_token=10000):
    messages = [
        {"role": "system", "content": "You are a journalist in a newsroom. A new event has occurred, you are given several headlines from competitors and an article snippet, and are given several tasks."}
    ]

    for article in articles:
        article_tokens = word_tokenize(article)
        if len(article_tokens) > max_token:
            article = ' '.join(article_tokens[:max_token])
        messages.append({"role": "user", "content": article})

    task_description = """
    Given the above news articles, complete the below two tasks:
    Task 1: Write down 5 central factual questions for the news event that most sources will likely answer. These questions and their answers should relate to the most important facts of the event. For example, for the US Presidential Election, the questions might be: Who won the election? What is the electoral college vote? What is the popular vote? What is the margin of victory? (each question should be up to 14 words)
    Task 2: Write down 15 opinion or prediction questions for the news event that most sources will likely answer uniquely. These questions and their answers should surface important points that news sources might analyze differently. For example, the questions might be: Who is more likely to win an election? Will there be a recession in 2023? What are the causes of the recession? (each question should be up to 14 words)
    In your answer, specify the task number explicitly (Task 1, Task 2) and use line breaks between tasks so that your report is structured.
    """
    messages.append({"role": "user", "content": task_description})

    return messages

# Function to load data from JSON or JSONL files
def load_json(filepath):
    with open(filepath, "r") as f:
        if filepath.endswith('jsonl'):
            return [json.loads(line) for line in f.readlines()]
        else:
            return json.load(f)

# Process a single event
def process_event(event, aid2article, questions, output_path):
    event_id = event['_id']
    event_aids = event['aids']
    this_questions = [q for q in questions if q['event_id'] == event_id]
    selected_question = this_questions[len(this_questions) // 2]
    top_5_clusters = selected_question['clusters'][:5]

    selected_articles = []
    for cluster in top_5_clusters:
        aids = [article['aid'] for article in cluster]
        article_contents = [aid2article[aid]['content'] for aid in aids if aid in aid2article]
        if article_contents:
            sorted_contents = sorted(article_contents, key=len)
            middle_article = sorted_contents[len(sorted_contents) // 2]
            truncated_article = "\n\n".join(middle_article.split("\n\n")[:10])
            selected_articles.append(truncated_article)
    
    prompt = format_prompt(selected_articles)
    prediction = response_API_with_retry(prompt, selected_articles)
    
    result = {
        "eid": event_id,
        "questions": prediction,
        "aids": event_aids
    }

    with open(output_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

def response_API_with_retry(prompt, articles):
    prediction = response_API(prompt)
    if prediction is None:
        for max_token in range(16000, 1, -1):
            prompt = format_prompt(articles, max_token=max_token)
            prediction = response_API(prompt)
            if prediction is not None:
                break
    return prediction

def main():
    questions = load_json("PATH/TO/QUESTIONS.json")
    events = load_json("PATH/TO/EVENTS.json")
    articles = load_json("PATH/TO/ARTICLES.json")

    # Create lookup dictionaries to access articles and events by ID
    aid2article = {a["_id"]: a for a in articles}
    eid2event = {e["_id"]: e for e in events}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for event in tqdm(events):
            futures.append(executor.submit(process_event, event, aid2article, questions, args.output_path))

        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
