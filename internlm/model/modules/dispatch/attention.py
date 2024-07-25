from internlm.core.parallel.comm.isp import auto_wrap_distributed_attention

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
    SUPPORT_FLASH2 = True
except ImportError:
    pass


@auto_wrap_distributed_attention
def flash_attn_wo_mask(
        query_states,
        key_states,
        value_states,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
):
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal)
    return attn_output


@auto_wrap_distributed_attention
def varlen_flash_attn(
        query_states,
        key_states,
        value_states,
        cumulative_len,
        max_seqlen,
        dropout_p=0.,
        causal=True,
):
    q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(
        0, 1), value_states.flatten(0, 1)
    attn_output = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cumulative_len,
        cumulative_len,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        return_attn_probs=False,
        causal=causal)
    attn_output = attn_output.unsqueeze(0)
    return attn_output
