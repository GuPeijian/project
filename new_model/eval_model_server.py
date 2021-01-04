import sys
sys.path.append("..")
from transformers import BertTokenizer
from UniLM import FuseModel
import re
import torch
import flask
import random

app = flask.Flask(__name__)
tokenizer = None
model = None


def load_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

def load_model(model_path):
    """Load model"""
    global model
    device = torch.device("cuda")
    model = FuseModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

def post_process(comment_ids):
    comment = tokenizer.decode(comment_ids)
    comment = re.sub(r"\[\S+\]", "", comment)
    comment = re.sub(r" ", "", comment)
    """
    if comment_ids[-1] == tokenizer.sep_token_id:
        # if truncate, cut the last sentence after the last punctuation
        comment = re.sub(r"[^。，！？,.?]", "", comment)
        if comment[-1] == "," or comment[-1] == "，":
            comment = comment[:-1]
    """
    return comment

def comment_generate(model,texts):
    user_ids = [tokenizer.convert_tokens_to_ids(
        '<user1>'), tokenizer.convert_tokens_to_ids('<user2>')]
    # process weibo
    text = texts[0]
    text = re.sub(r"\[\S+\]", "", text)
    text = re.sub(r"(\S)\1{3,}", r"\1\1", text)
    text_ids = tokenizer.encode(
        text, max_length=300, truncation=True, add_special_tokens=True)
    # add token
    text_ids = text_ids + [user_ids[0]]
    num_turns = len(texts) - 1
    max_history_turns = 3
    if num_turns > 0:
        # only include previous comments num of max_history_turns
        history_comments = texts[-(max_history_turns + 1):][1:]

        for i, comment in enumerate(history_comments):
            # encode
            comment_ids = tokenizer.encode(
                comment, max_length=50, truncation=True, add_special_tokens=False)
            # add token
            comment_ids = comment_ids + \
                          [tokenizer.sep_token_id] + [user_ids[(i + 1) % 2]]
            # concat
            text_ids = text_ids + comment_ids

    text_ids = torch.tensor([text_ids])
    device = torch.device("cuda")
    input_ids = text_ids.to(device)
    text_len = input_ids.size()[1]
    #sentiment = torch.tensor([[sentiment]]).to(device)

    #generate comments and control length with prob 0.2
    n=0
    while n<10:
        comment_ids = model.generate(input_ids, topp=0.90, no_repeat_ngram_size=3, do_sample=True, max_length=text_len + 50,
                                     min_length=text_len + 7,temperature=0.8, sentiment_input=None)
        comment_ids = comment_ids[0][text_len:]
        if comment_ids.size()[0]>30:
            prob=random.uniform(0,1)
            if prob>=0.8:
                break
        else:
            break
        n+=1

    return comment_ids

def comment_generate_keywords(model,texts,keywords):
    user_ids = [tokenizer.convert_tokens_to_ids(
        '<user1>'), tokenizer.convert_tokens_to_ids('<user2>')]
    keywords_ids=tokenizer.convert_tokens_to_ids("<keywords>")
    # process weibo
    text = texts[0]
    text = re.sub(r"\[\S+\]", "", text)
    text = re.sub(r"(\S)\1{3,}", r"\1\1", text)
    text_ids = tokenizer.encode(
        text, max_length=280, truncation=True, add_special_tokens=True)
    # add dialog tokens
    num_turns = len(texts) - 1
    max_history_turns = 3
    i=-1
    if num_turns > 0:
        # only include previous comments num of max_history_turns
        history_comments = texts[-(max_history_turns + 1):][1:]

        for i, comment in enumerate(history_comments):
            # encode
            comment_ids = tokenizer.encode(
                comment, max_length=50, truncation=True, add_special_tokens=False)
            # add token
            comment_ids = [user_ids[i % 2]]+ comment_ids + \
                          [tokenizer.sep_token_id]
            # concat
            text_ids = text_ids + comment_ids
    #add keywords
    if keywords!=[]:
        for keyword in keywords:
            keyword_id = tokenizer.encode(keyword, add_special_tokens=False)
            keyword_id += [tokenizer.sep_token_id]
            text_ids += keyword_id
    #add first token for generation
    if i==-1: # no dialog
        text_ids+=[user_ids[0]]
    else:
        text_ids+=[user_ids[(i+1)%2]]

    text_ids = torch.tensor([text_ids])
    device = torch.device("cuda")
    input_ids = text_ids.to(device)
    text_len = input_ids.size()[1]
    #generate comments and control length with prob 0.2
    n=0
    while n<10:
        comment_ids = model.generate(input_ids, topp=0.90, no_repeat_ngram_size=3, do_sample=True, max_length=text_len + 50,
                                     min_length=text_len + 7,temperature=0.8, sentiment_input=None)
        comment_ids = comment_ids[0][text_len:]
        if comment_ids.size()[0]>30:
            prob=random.uniform(0,1)
            if prob>=0.8:
                break
        else:
            break
        n+=1

    return comment_ids

@app.route("/generation", methods=["POST"])
def generation():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.json.get("text"):
            texts = flask.request.json["text"]
            comment_ids=comment_generate(model,texts)
            comment = post_process(comment_ids)

            data["comment"] = comment
            data["success"] = True

    return flask.jsonify(data)

@app.route("/keywords", methods=["POST"])
def keywords():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.json.get("text"):
            texts = flask.request.json["text"]
            keywords=flask.request.json["keywords"]
            comment_ids = comment_generate_keywords(model, texts,keywords)
            comment = post_process(comment_ids)

            data["comment"] = comment
            data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    #model_path="../all_model/hxj_and_news"
    model_path="../all_model/keywords_model_raw"
    load_tokenizer(model_path)
    load_model(model_path)
    app.run(host='127.0.0.1',
            port=2000,
            debug=True)
