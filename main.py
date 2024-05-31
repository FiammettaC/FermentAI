from flask import Flask, render_template, request
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM, FalconForCausalLM
from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
import pandas as pd

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())

def semantic_search(asked_question):
    model_search = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    df = pd.read_excel('fermentation_qa.xlsx')
    # with open('data_gmp2.json') as f:
    #     data = json.load(f)

    similarity = []
    answers = []
    embedding_1 = model_search.encode(asked_question, convert_to_tensor=True)
    for idx, (question, answer) in enumerate(zip(df.question, df.answer)):
        embedding_2 = model_search.encode(question, convert_to_tensor=True)
        # Similarity of two documents
        similarity.append(util.pytorch_cos_sim(embedding_1, embedding_2)[0].cpu().numpy())
        #answers.append(answer)
    most_similar = similarity.index(max(similarity))    
    return df.iloc[most_similar]['answer']

def predict(question):
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')       
    context = semantic_search(question)
    min_length = int(len(context.split())*0.8)
    #input = f"INSTRUCTION: Please answer the QUESTION using the provided CONTEXT. QUESTION: {question} CONTEXT: {context}"# INSTRUCTIONS: GENERATE ANSWER TO THE QUESTION BASED ON THE CONTEXT GIVEN"
    input = f"Based on the CONTEXT, Please answer this QUESTION: QUESTION: {question} CONTEXT: {context}"
    
    encoded_input = tokenizer([input],
                                 return_tensors='pt',
                                 max_length=1024,
                                 truncation=False).to(device)
    model = model.to(device)
    output = model.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask,
                            num_beams=5,
                            #num_return_sequences=10,
                            max_length=int(len(context.split()))+1,       #
                            min_new_tokens=min_length,                    #
                            penalty_alpha=0.6,                            #
                            top_k=4,                                      #
                            do_sample=True,                               #
                            )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

@app.route("/")
def home():    
    return render_template("index.html")

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response = predict(userText)  
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)