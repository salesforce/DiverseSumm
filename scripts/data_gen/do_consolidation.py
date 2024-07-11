# adapted from https://github.com/salesforce/discord_questions/blob/d7cbd514895bdfbb54782645909eea70fe1435b3/dq_pipeline.py
import json
from tqdm import tqdm
import argparse
from model_consolidation import ConsolidationModel

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--generated_question_path', type=str, required=True, help="The file produced by gpt_qg.py.")
parser.add_argument('--generated_answer_path', type=str, required=True, help="The file produced by gpt_qa.py.")
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

# Initialize the consolidation model
model = ConsolidationModel(model_card='Salesforce/qa_consolidation')

# Function to load data from JSON or JSONL files
def load_json(filepath):
    with open(filepath, "r") as f:
        if filepath.endswith('jsonl'):
            return [json.loads(line) for line in f.readlines()]
        else:
            return json.load(f)

# Load all necessary data
questions = load_json("PATH/TO/QUESTIONS.json")
events = load_json("PATH/TO/EVENTS.json")
articles = load_json("PATH/TO/ARTICLES.json")
questions_with_events = load_json(args.generated_question_path)
generated_answers = load_json(args.generated_answer_path)

# Create lookup dictionaries
aid2article = {a["_id"]: a for a in articles}
eid2event = {e["eid"]: e for e in questions_with_events}

# Function to consolidate answers for a single question-event pair
def consolidate_answers(event, generated_answers, model):
    answers_for_all_questions = []
    this_generated_questions = event['questions']
    
    assert len(generated_answers) == len(this_generated_questions), (len(generated_answers), len(this_generated_questions))
    
    for question, articles_answers in zip(this_generated_questions, generated_answers):
        assert question == articles_answers['question']
        aids = articles_answers['aids']
        articles_answers = articles_answers['answers']
        
        assert len(articles_answers) == len(aids), (len(articles_answers), len(aids))
        
        valid_articles = []
        for aid, answers in zip(aids, articles_answers):
            for answer in answers:
                valid_articles.append({'answer': answer, 'aid': aid})
        
        try:
            answer_groups = model.consolidate(question=question, paragraphs=valid_articles)
        except Exception as e:
            print(f"Error consolidating answers: {e}")
            answer_groups = []
        
        result = {
            'eid': event['eid'],
            'question': question,
            'answer_groups': answer_groups
        }
        answers_for_all_questions.append(result)
    
    return answers_for_all_questions

# Main processing loop
def main():
    for event, this_generated_answers in tqdm(zip(questions_with_events, generated_answers), total=len(generated_answers)):
        answers_for_all_questions = consolidate_answers(event, this_generated_answers, model)
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(answers_for_all_questions) + '\n')

if __name__ == "__main__":
    main()