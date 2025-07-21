from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

import torch


def load_model():
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# def load_model():
#     model_id = "google/gemma-3-1b-it"

#     quant_config = BitsAndBytesConfig(load_in_8bit=True)

#     model = Gemma3ForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=quant_config
#     ).eval()

#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     return model, tokenizer


# def get_top_n_next_words(sequence, model, tokenizer, n=3):
#     print(f"[Model Input] Sequence: '{sequence}'")

#     input_ids = tokenizer.encode(tokenizer.bos_token + " " + sequence, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits[:, -1, :]  # focus on last token
#         probs = torch.softmax(logits, dim=-1)
#         top_probs, top_indices = torch.topk(probs, n)

#         next_words = [tokenizer.decode(idx.item()).strip()
#                       for idx in top_indices[0]]
#         probabilities = top_probs[0].tolist()

#     for word, prob in zip(next_words, probabilities):
#         print(f"[Prediction] '{word}' with probability {prob:.4f}")

#     return list(zip(next_words, probabilities))


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (nucleus)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least min_tokens_to_keep tokens
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('inf')

    return logits

def get_top_n_next_words(sequence, model, tokenizer, n=3):
    print(f"[Model Input] Sequence: '{sequence}'")

    input_ids = tokenizer.encode(tokenizer.bos_token + " " + sequence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # focus on last token

        # Apply temperature scaling
        temperature = 0.7
        logits = logits / temperature

        # Apply top-k filtering using custom function
        filtered_logits = top_k_top_p_filtering(logits, top_k=n)

        probs = torch.softmax(filtered_logits, dim=-1)

        # Get top n from filtered logits
        top_probs, top_indices = torch.topk(probs, n)

        next_words = [tokenizer.decode(idx.item()).strip() for idx in top_indices[0]]
        probabilities = top_probs[0].tolist()

    for word, prob in zip(next_words, probabilities):
        print(f"[Prediction] '{word}' with probability {prob:.4f}")

    return list(zip(next_words, probabilities))
