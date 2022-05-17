#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from representation_generator_utils import check_folder_exists, extract_paraphrase_data
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as st

class Model_sent:
    def __init__(self, model):
        self.model_name = model
        if model=='bert-base':
            self.model = "bert-base-cased"
        if model=='bert-large':
            self.model = "bert-large-cased"
        if model=='roberta-base':
            self.model = "roberta-base"
        if model=='roberta-large':
            self.model = "roberta-large"
        if model=='gpt2':
            self.model = "gpt2"    


        
    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model,cache_dir="huggingface_cache")  
        self.model_out = AutoModel.from_pretrained(self.model, output_hidden_states=True,cache_dir="huggingface_cache")
        return self.tokenizer,self.model_out
    
    def tok(self,sent):
        encoded = self.tokenizer.encode_plus(text=sent,add_special_tokens=True,return_tensors = 'pt')
        return encoded['input_ids']
    
    def word_repr(self,word,pooler=False):
        pooler_output = None
        
        outputs = self.model_out(word)

        if not pooler:
            hidden_states = outputs.hidden_states
            last_hidden = outputs.last_hidden_state

        if pooler: #FOR BERT OUTPUTS
            pooler_output = outputs[1] #shape=[1,768] --> hidden state corresponding to the first token
            last_hidden = outputs[0] # shape=[1,w,768]...final output       
            hidden_states = outputs[2] #len=13....0th layer output is representation from embedding layer
        
        return last_hidden,pooler_output,hidden_states
    
    def get_representations(self,n_layers,sent):
        
        tok = self.tok(sent) #shape=[1,w]
        
        final_layer_tokens, pooler, hidden_states = self.word_repr(tok,pooler=True) 
            
        Mean = []
        
        for lyr in range(n_layers):
            layer_rep = hidden_states[lyr][0]
                
            mean_representation = layer_rep.mean(axis=0).cpu().detach().numpy()
            mean_rep = np.array(mean_representation,dtype=np.float64)
            Mean.append(mean_rep)
 

        return Mean


    
def model_init(model_name):
    model = Model_sent(model_name)
    tok, out = model.init_model()
    return model

def cosine_sim(r1,r2):
    return distance.cosine(r1,r2)


def get_representations_sentence_pairs(n_layers, model, sentence_pairs):
    Sim = []
    for p in tqdm(range(len(sentence_pairs))):
        pair = sentence_pairs[p]
        Sm = []
        rep1 = model.get_representations(n_layers,pair[0])
        rep2 = model.get_representations(n_layers,pair[1])        
        for layer in range(len(rep1)):
            Sm.append(cosine_sim(rep1[layer],rep2[layer]))
        Sim.append(Sm)
    print(pair)
    return np.array(Sim)


def confidence(vals):
    return st.t.interval(
        alpha=0.95,
        df=len(vals) - 1,
        loc=np.mean(vals),
        scale=st.sem(vals)
    )


def plotter(label,n_layers,dataset):
    PLTARGS = dict(
        capsize=3, capthick=2,
        ms=10, marker=".",
        elinewidth=1
    )
    ys = []
    for cols in range(dataset.shape[1]):
        ys.append(dataset[:,cols]) 
    cs = [confidence(y) for y in ys]
    plot_mat = [np.average(y) for y in ys]
    
    yerr = [(x[1] - x[0]) / 2 for x in cs]
    plt.errorbar(
        list(range(n_layers)),
        [np.average(y) for y in ys],
        label=label,
        yerr=yerr,
        **PLTARGS
    )

def calc_model_wise(model_name):
    model = model_init(model_name)
    n_layers=13
    if model_name=="bert-large" or model_name=="roberta-large":
        n_layers=25
    
    print("processing not paraphrase cases")
    case0 = get_representations_sentence_pairs(n_layers,model,sent0)
    print("processing paraphrase cases")
    case1 = get_representations_sentence_pairs(n_layers,model,sent1)
    plotter("Normal",n_layers,case0)
    plotter("Paraphrase",n_layers,case1)

    plt.ylabel("Average cosine similarity")
    plt.xlabel("Layer")
    plt.title(model_name+": Cosine distance among sentence pairs")
    plt.tight_layout(pad=0.1)
    plt.legend(ncol=2)    
    
    plt.savefig("Sentence_cosine_distance_"+model_name)

    plt.clf()
    plt.close()

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-d", "--data")
    args = args.parse_args()

    data_location = args.data

    rep_folder =  os.path.join(os.getcwd(),"Representations")
    check_folder_exists(rep_folder)
    
    data_folder = os.path.join(os.getcwd(),os.path.join("data","paraphrase_identification"))
    sent0, sent1 = extract_paraphrase_data(data_folder) 

    # calc_model_wise("bert-base")
    # calc_model_wise("bert-large")    
    # calc_model_wise("roberta-base")
    # calc_model_wise("roberta-large")
    calc_model_wise("gpt2")
        
    
   