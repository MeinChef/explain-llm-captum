# import torch
from captum.attr import ShapleyValueSampling, LLMAttribution, TextTemplateInput, ProductBaselines
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup():
    # model is 2gb, is this too much?
    # takes a good bit of time to execute on my gpu, and since other people might not have access to gpu, this is not viable

    # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 300MB model, script takes 20s to run on GPU
    # also distilgpt is more biased, this should be great
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return model, tokenizer

def do_stuff(model, tokenizer):
    now = time.time()
    svs = ShapleyValueSampling(model)
    baselines = ProductBaselines(
            {
                ("name" , "pronoun"): [("Sarah", "Her"), ("John", "His")],
                "city": ["Seattle", "Boston"],
                "state": ["WA", "MA"],
                "occupation": [" doctor", "engineer", "teacher", "technician", "plumber"]
            }
        )
    llm_attr = LLMAttribution(svs, tokenizer)
    inp = TextTemplateInput(
            "{name} lives in {city} , {state} and is a {occupation}. {pronoun} personal interests include",
            {
                "name"      : "Dave", 
                "city"      : "Palm Beach",
                "state"     : "FL",
                "occupation": "lawyer",
                "pronoun"   : "His"
            },
            baselines = baselines ,
        )
    attr_result = llm_attr.attribute(inp, target= "playing golf, hiking, and cooking.")
    print(f"Time: {time.time()-now}s")
    return attr_result

if __name__ == "__main__":
    model, tokenizer = setup()
    res = do_stuff(model, tokenizer)
    res.plot_token_attr(show = True)