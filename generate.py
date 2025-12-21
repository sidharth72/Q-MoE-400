import torch

from generation_config import GenerationConfig, generate, get_tokenizer
from Q_MoE_400_torch import count_parameters, load_model_from_checkpoint


CHECKPOINT_PATH = "./moe_torch_state_dict_90000.pt"

PROMPT = """Hello I'm a language model,"""

MAX_NEW_TOKENS = 100
TEMPERATURE = 1
DO_SAMPLE = True
TOP_K = 50
TOP_P = 0.95
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50256


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(CHECKPOINT_PATH, device=device)

    # total_params, trainable_params = count_parameters(model)
    # print(f"total parameters: {total_params:,}")
    # print(f"trainable parameters: {trainable_params:,}")

    encode, decode = get_tokenizer(allow_download=False)
    prompt_ids = encode(PROMPT)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )

    out_ids = generate(model=model, input_ids=input_ids, cfg=cfg)
    out_text = decode(out_ids[0].tolist())
    print(out_text)


if __name__ == "__main__":
    main()
