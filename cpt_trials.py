# import torch
from captum.attr import ShapleyValueSampling, LLMAttribution, TextTemplateFeature, ProductBaselines

from transformers import AutoModelForCausalLM, AutoTokenizer

def setup():
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return model, tokenizer

def do_stuff(model, tokenizer):
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
    inp = TextTemplateFeature(
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
    
    return attr_result

if __name__ == "__main__":
    model, tokenizer = setup()
    print(do_stuff(model, tokenizer))