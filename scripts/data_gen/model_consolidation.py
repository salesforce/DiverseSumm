# copied from https://github.com/salesforce/discord_questions/blob/master/model_consolidation.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import networkx as nx, numpy as np, community, torch, tqdm

class ConsolidationModel:
    def __init__(self, model_card, model_file=None, device="cuda"):
        self.model_card = model_card
        self.model_file = model_file
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_card).to(self.device)

        if model_file is not None:
            loaded_dict = torch.load(model_file)
            print(self.model.load_state_dict(loaded_dict))
        self.model.eval()

    def get_logits(self, texts):
        input_ids = [torch.LongTensor(self.tokenizer.encode(text)) for text in texts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(self.device)

        self.model.eval()
        with torch.no_grad():
            model_outs = self.model(input_ids=input_ids)
            logits = model_outs["logits"]
        return logits

    def score_batch(self, questions, answers1, answers2, contexts1):
        texts = ["%s <sep> %s <sep> %s" % (q, a1, a2) for q, a1, a2 in zip(questions, answers1, answers2)]
        with torch.no_grad():
            logits = self.get_logits(texts)
        return {"scores": logits[:, 0].tolist()}

    def score(self, questions, answers1, answers2, contexts1, contexts2=None, batch_size=32, progress=False):
        N = len(questions)
        scores = []
        ite = range(0, N, batch_size)
        if progress and len(ite) > 1:
            ite = tqdm.tqdm(ite)
        for i in ite:
            batch_q = questions[i:i+batch_size]
            batch_a1 = answers1[i:i+batch_size]
            batch_a2 = answers2[i:i+batch_size]
            batch_c1 = contexts1[i:i+batch_size]
            batch_scores =  self.score_batch(batch_q, batch_a1, batch_a2, batch_c1)["scores"]
            for idx, a1, a2 in zip(range(len(batch_a1)), batch_a1, batch_a2):
                if a1 == a2:
                    batch_scores[idx] = 5.0
            scores += batch_scores
        return {"scores": scores}

    def compare_batch(self, question, p1s, p2s):
        scores = []

        to_do = [i for i in range(len(p1s)) if p1s[i]["answer"] != p2s[i]["answer"]]
        questions = [question] * len(to_do)
        contexts = [""] * len(to_do)
        answers1 = [p1s[i]["answer"] for i in to_do]
        answers2 = [p2s[i]["answer"] for i in to_do]
        idx2score = {}
        if len(to_do) > 0:
            non_triv_scores = self.score_batch(questions, answers1, answers2, contexts1=contexts)["scores"]
            idx2score = {i: non_triv_scores[idx] for idx, i in enumerate(to_do)}
        scores = [idx2score.get(i, 5.0) for i in range(len(p1s))]
        return {"scores": scores}

    def compare(self, question, p1s, p2s, batch_size=512, progress=True):
        N = len(p1s)
        scores = []
        ite = range(0, N, batch_size)
        if progress and len(ite) > 1:
            ite = tqdm.tqdm(ite)
        for i in ite:
            batch_p1s = p1s[i:i+batch_size]
            batch_p2s = p2s[i:i+batch_size]
            batch_scores = self.compare_batch(question, batch_p1s, batch_p2s)
            scores.extend(batch_scores["scores"])
        return {"scores": scores}

    def build_graph(self, question, paragraphs, thresh=2.75):
        paragraph_pairs = [{"p1": p1, "p1_idx": i, "p2": p2, "p2_idx": j} for i, p1 in enumerate(paragraphs) for j, p2 in enumerate(paragraphs) if i != j]
        p1s = [p["p1"] for p in paragraph_pairs]
        p2s = [p["p2"] for p in paragraph_pairs]

        scores = self.compare(question, p1s, p2s, batch_size=512, progress=False)["scores"]
        weight_matrix = np.zeros((len(paragraphs), len(paragraphs)))
        for p, s in zip(paragraph_pairs, scores):
            p["score"] = s
            weight_matrix[p["p1_idx"], p["p2_idx"]] += s / 2
            weight_matrix[p["p2_idx"], p["p1_idx"]] += s / 2
        weight_matrix = weight_matrix > thresh
        G = nx.from_numpy_matrix(weight_matrix)
        return G

    def consolidate(self, question, paragraphs):
        G = self.build_graph(question, paragraphs)
        sub_parts = community.best_partition(G, randomize=False)
        sub_comps = {}
        for top, c in sub_parts.items():
            if c not in sub_comps:
                sub_comps[c] = set([])
            sub_comps[c].add(top)

        groups = sorted(sub_comps.values(), key=len, reverse=True)
        answer_groups = []
        for group in groups:
            answer_groups.append([paragraphs[i]["answer"] for i in group])
        return answer_groups