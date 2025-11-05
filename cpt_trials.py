import torch
from captum.attr import ShapleyValueSampling, LLMAttribution, TextTemplateInput, ProductBaselines, Saliency
from captum.metrics import infidelity, sensitivity_max, infidelity_perturb_func_decorator
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def setup():
    # model is 2gb, is this too much?
    # takes a good bit of time to execute on my gpu, and since other people might not have access to gpu, this is not viable

    # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 300MB model, script takes 20s to run on GPU, and 27 on my laptop - nice
    # also distilgpt is more biased, this should be great
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return model, tokenizer

@infidelity_perturb_func_decorator
def perturbfn(inputs):
    """
    Generic perturbation: return inputs mixed with baselines according to noise.
    - inputs, baselines: tensors/arrays/embeddings (or anything that supports elementwise ops)
    - noise: mask or values in [0,1] with shape broadcastable to inputs
    Example behavior: noise==1 -> use baseline, noise==0 -> keep input.
    """
    # return inputs * (1 - noise) + baselines * noise
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()  
    return noise, inputs - noise

def do_stuff(model, tokenizer):
    now = time.time()
    svs = ShapleyValueSampling(model)
    # svs = Saliency(model)
    # take feature ablation as an additional
    # contrast heatmaps

    # make baselines more easily editable
    baselines = ProductBaselines(
            {
                ("name" , "pronoun"): [("Sarah", "Her"), ("John", "His")],
                "city": ["Seattle", "Boston"],
                "state": ["WA", "MA"],
                "occupation": [" doctor", "engineer", "teacher", "technician", "plumber"]
            }
        )
    
    values = {
        "name"      : "Dave", 
        "city"      : "Palm Beach",
        "state"     : "FL",
        "occupation": "lawyer",
        "pronoun"   : "His"
    }
    llm_attr = LLMAttribution(svs, tokenizer)
    
    inp = TextTemplateInput(
            "{name} lives in {city}, {state} and is a {occupation}. {pronoun} personal interests include",
            values,
            baselines = baselines,
        )
    attr_result = llm_attr.attribute(inp, target = "playing golf, hiking, and cooking.")
    # inputs = torch.tensor(tokenizer(inp.to_model_input())["input_ids"])


    # Adapter: sensitivity_max will pass torch.Tensor inputs; convert token ids -> decoded text
    # and build a TextTemplateInput for LLMAttribution.attribute.
    # def explanation_from_token_tensor(token_tensor_tuple, **kwargs):
    #     # Unpack the tuple - first element contains the tensor
    #     token_tensor = token_tensor_tuple[0]
        
    #     # Convert tensor to list of token ids
    #     if token_tensor.dim() > 1:
    #         ids = token_tensor[0].tolist()  # take first batch if batched
    #     else:
    #         ids = token_tensor.tolist()  # use directly if unbatched
            
    #     decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
    #     tt = TextTemplateInput(
    #         template = decoded,
    #         values = values,
    #         baselines = baselines
    #     )
    #     return llm_attr.attribute(tt, **kwargs)

    # print(type(inputs))
    # attr_result = svs.attribute(inputs)
    # infd = infidelity(
    #     forward_func = model,
    #     perturb_func = perturbfn ,
    #     inputs = inputs, 
    #     attributions = attr_result
    # )
    # # sens = sensitivity_max(
    # #     explanation_func = explanation_from_token_tensor,
    # #     inputs = inputs,
    # #     target = "playing golf, hiking, and cooking."
    # # )
    print(
        "Input Tokens: ",
        attr_result.input_tokens,
        "\nOutput Tokens: ",
        attr_result.output_tokens,
        "\nSeq Attr: ",
        attr_result.seq_attr,
        "\nToken Attr: ",
        attr_result.token_attr,
        "\nSeq Dict: ",
        attr_result.seq_attr_dict
    )
    print(f"Time: {time.time()-now}s")

    # TODO:
    # few shot (figure 7)
    # make pretty and explain stuff
    return None

if __name__ == "__main__":
    model, tokenizer = setup()
    res = do_stuff(model, tokenizer)
    res.plot_token_attr(show = True)