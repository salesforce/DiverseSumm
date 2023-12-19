# Embrace Divergence for Richer Insights: A Multi-document Summarization Benchmark and a Case Study on Summarizing Diverse Information from News Articles



Paper link: [Arxiv](https://arxiv.org/abs/2309.09369) 
## Abstract
Previous research in multi-document news summarization has typically concentrated on collating information that all sources agree upon. However, to our knowledge, the summarization of diverse information dispersed across multiple articles about an event has not been previously investigated. The latter imposes a different set of challenges for a summarization model. In this paper, we propose a new task of summarizing diverse information encountered in multiple news articles encompassing the same event. To facilitate this task, we outlined a data collection schema for identifying diverse information and curated a dataset named DiverseSumm. The dataset includes 245 news stories, with each story comprising 10 news articles and paired with a human-validated reference. Moreover, we conducted a comprehensive analysis to pinpoint the position and verbosity biases when utilizing Large Language Model (LLM)-based metrics for evaluating the coverage and faithfulness of the summaries, as well as their correlation with human assessments. We applied our findings to study how LLMs summarize multiple news articles by analyzing which type of diverse information LLMs are capable of identifying. Our analyses suggest that despite the extraordinary capabilities of LLMs in single-document summarization, the proposed task remains a complex challenge for them mainly due to their limited coverage, with GPT-4 only able to cover less than 40% of the diverse information on average.



## DiverseSumm
We release the data for our DiverseSumm dataset in the `data` folder.


### Data Format

Below shows the format of an instance in DiverseSumm:
```
{
  "eid" : "...",
  "articles": [
    {
      "aid": "...",
      "title: "...",
      "url": "...",
      "domain": "...",
      "content": "..."
    },
  ],
  "question_answers":[
    {
      "question": "...",
      "answer_groups":[
        [
          {
            "aid": "...",
            "answer": "..."
          },
        ],
        [
          ...
        ]
        
      ]
    },
  ]

}
```
### Fields
* `eid`:  A unique string identifier for the event.
* `articles`: A list of articles, each of which is represented as a dictionary.
  * `aid`: An identifier for each individual article within an event.
  * `title`: The headline or title of the given article.
  * `url`: The original link to the online version of the article.
  * `domain`: The website or news source where the article was originally published.
  * `content`: The full text content of the article.
* `question_answers`: A list of dictionary entries each containing a question and various groups of answers, relating to that question, from different articles.
  * `question`: A specific question about the event that the result in diverse answers across different articles.
  * `answer_groups`: A list of lists where each sub-list represents a group of answers. Each answer may come from a different article about the event. 
    * `answer`: The text of the answer giving information about the event. It answers the question from the same dictionary entry and is extracted from the article with the corresponding `aid`.

## Code
Coming soon.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{huang2023embrace,
  title     = "Embrace Divergence for Richer Insights: A Multi-document Summarization Benchmark and a Case Study on Summarizing Diverse Information from News Articles",
  author    = "Kung-Hsiang Huang and Philippe Laban and Alexander R. Fabbri and Prafulla Kumar Choubey and Shafiq Joty and Caiming Xiong and Chien-Sheng Wu",
  year = "2023",
  eprint={2309.09369},
  archivePrefix={arXiv},

}
```
