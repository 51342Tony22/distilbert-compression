# 用于构建 DistilBERT 模型并执行 L1 注意力头结构化剪枝

from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification
)
from runtime import from_pretrained_local_first


def prune_lowest_l1_attention_heads(
    model,
    heads_to_prune_per_layer=2,
):
    """根据 L1 范数物理删除重要性最低的注意力头"""

    if heads_to_prune_per_layer <= 0:
        return {}

    heads_to_prune = {}

    for layer_index, layer in enumerate(
        model.distilbert.transformer.layer
    ):
        attention = layer.attention
        head_dimension = attention.dim // attention.n_heads
        head_scores = []

        for head_index in range(attention.n_heads):
            start = head_index * head_dimension
            end = start + head_dimension
            #计算每个头的权重，减去最低的两个
            score = (
                attention.q_lin.weight[start:end].abs().sum()
                + attention.k_lin.weight[start:end].abs().sum()
                + attention.v_lin.weight[start:end].abs().sum()
                + attention.out_lin.weight[:, start:end].abs().sum()
            )

            head_scores.append((float(score.item()), head_index))

        head_scores.sort(key=lambda item: item[0])
        heads_to_prune[layer_index] = [
            head_index
            for _, head_index in head_scores[:heads_to_prune_per_layer]
        ]

    model.prune_heads(heads_to_prune)
    return heads_to_prune


#读取固有模型
def build_model(
    num_labels=3,
    num_hidden_layers=6,
    dropout=0.2,
    attention_dropout=0.2,
    heads_to_prune_per_layer=2,
):
    config = from_pretrained_local_first(
        DistilBertConfig,
        num_labels=num_labels
    )

    config.n_layers = num_hidden_layers

    config.dropout = dropout
    config.attention_dropout = attention_dropout
    config.seq_classif_dropout = dropout

    model = from_pretrained_local_first(
        DistilBertForSequenceClassification,
        config=config,
        ignore_mismatched_sizes=True
    )

    prune_lowest_l1_attention_heads(
        model,
        heads_to_prune_per_layer=heads_to_prune_per_layer,
    )

    return model


def count_trainable_parameters(model):
    """统计模型中可训练参数的数量"""

    total_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params


def print_model_info(model):
    """打印模型的基础结构信息。"""

    num_layers = len(
        model.distilbert.transformer.layer
    )

    total_params = count_trainable_parameters(model)
    heads_per_layer = [
        layer.attention.n_heads
        for layer in model.distilbert.transformer.layer
    ]

    print("===== Model Structure Information =====")
    print(f"Transformer layers: {num_layers}")
    print(f"Attention heads per layer: {heads_per_layer}")
    print(f"Pruned heads: {model.config.pruned_heads}")
    print(f"Trainable parameters: {total_params:,}")
