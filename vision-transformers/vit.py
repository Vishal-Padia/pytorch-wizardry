import math
import torch
from torch import nn

class NewGELUActivation(nn.Module):
    """
    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    """
    convert the image into patches and project them on a vector space
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.image_size = config["image_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # calculate num of patches from image size and patch size
        # num of patches = H.W / P^2
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # create a projection layer to convert the image into patches
        # and project each patch into a vector of size hidden_size
        self.projections = nn.Conv2d(in_channels=self.num_channels, out_channels=self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_channels, hidden_size)
        x = self.projections(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    """
    Combine those patch embedding with class token and position embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        # create a learnable param [CLS]
        # similar to bert, this [CLS] token is added to beginning of the input sequence to classify the whole sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # create position embeddings for the [CLS] token and patch embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, out=self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # expand [CLS] token to batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # add the cls tokens to the beginning of the input sequence
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x

class AttentionHead(nn.Module):
    """ A single attention head"""
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # create query, key, value
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.key(x)

        # calculate attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(input=attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # calculate attention output
        attention_output = torch.matmul(attention_probs, value)

        return (attention_output, attention_probs)

class MultiHeadAttention(nn.Module):
    """ Multi Head Attention Module """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # the attention head size is the hidden size divided by number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # whether or not to use bias for q,k,v
        self.qkv_bias = config["qkv_bias"]

        # create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=config["attention_probs_dropout_prob"],
                bias=True
            )
            self.heads.append(head)

        # a linear to project attention output back to hidden size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # calculate attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]

        # merge attention output from each heads
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)

        # project merged attention output to hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """ A multi-layer perceptron module """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    """ A single transformer block """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)

        # skip connection
        x = x + attention_output

        # feed forward neural network
        mlp_output = self.mlp(self.layernorm_2(x))

        # skip connection
        x = x + mlp_output

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)

class Encoder(nn.Module):
    """ The transformer encoder module """
    def __init__(self, config):
        super().__init__()
        # create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)

class ViTForClassification(nn.Module):
    """ The ViT model for classification """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        # create embedding module
        self.embedding = Embeddings(config)

        # Create the transformer encoder module
        self.encoder = Encoder(config)

        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
