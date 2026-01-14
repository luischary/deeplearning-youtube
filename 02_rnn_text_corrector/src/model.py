import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import PAD_TOKEN


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        h = torch.zeros(
            (2 * self.num_layers, x.shape[0], self.hidden_size), device=x.device
        )
        c = torch.zeros(
            (2 * self.num_layers, x.shape[0], self.hidden_size), device=x.device
        )

        x, (h, c) = self.rnn(x, (h, c))  # [B, L, 2 * H]
        return x, (h, c)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, h=None, c=None):
        x = self.embedding(tokens)

        if h is None or c is None:
            h = torch.zeros(
                (self.num_layers, x.shape[0], self.hidden_size), device=x.device
            )
            c = torch.zeros(
                (self.num_layers, x.shape[0], self.hidden_size), device=x.device
            )

        x, (h, c) = self.rnn(x, (h, c))  # [B, L, H]
        predictions = self.fc_out(x)  # [B, L, Vocab]

        return predictions, (h, c)


class Corretor(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_encoder,
        hidden_decoder,
        num_layers_encoder,
        num_layers_decoder,
        dropout=0.0,
    ):
        super(Corretor, self).__init__()

        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            hidden_encoder,
            num_layers=num_layers_encoder,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size,
            embed_dim,
            hidden_decoder,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )

        self.proj_h = nn.Linear(
            2 * hidden_encoder * num_layers_encoder, hidden_decoder * num_layers_decoder
        )
        self.proj_c = nn.Linear(
            2 * hidden_encoder * num_layers_encoder, hidden_decoder * num_layers_decoder
        )

    def project_memory(self, h, c):
        h = h.transpose(0, 1)  # [B, 2 * n_layers, H]
        h = h.reshape((h.shape[0], -1))  # [B, 2 * H * n_layers]
        h = self.proj_h(h)  # [B, H * n_layers]
        h = h.reshape(
            (h.shape[0], self.decoder.num_layers, self.decoder.hidden_size)
        )  # [B, n_layers, H]
        h = h.transpose(0, 1).contiguous()  # [n_layers, B, H]

        c = c.transpose(0, 1)  # [B, 2 * n_layers, H]
        c = c.reshape((c.shape[0], -1))  # [B, 2 * H * n_layers]
        c = self.proj_c(c)  # [B, H * n_layers]
        c = c.reshape(
            (c.shape[0], self.decoder.num_layers, self.decoder.hidden_size)
        )  # [B, n_layers, H]
        c = c.transpose(0, 1).contiguous()  # [n_layers, B, H]

        return h, c

    def forward(self, src_tokens, tgt_tokens, h=None, c=None):
        if h is None or c is None:
            encoder_outputs, (h, c) = self.encoder(src_tokens)
            # shape da memoria [2 * num_layers, B, H_Enc]
            # queremos [num_layers, B, H_Dec]
            h, c = self.project_memory(h, c)

        outputs, (h, c) = self.decoder(tgt_tokens, h, c)
        return outputs, (h, c)

    def train_step(self, batch):
        src_tokens, dec_tokens, trg_tokens = batch

        logits, (h, c) = self.forward(src_tokens, dec_tokens)
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            trg_tokens.reshape((-1,)),
            ignore_index=PAD_TOKEN,
        )

        return loss

    def valid_step(self, batch):
        src_tokens, dec_tokens, trg_tokens = batch

        logits, (h, c) = self.forward(src_tokens, dec_tokens)
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            trg_tokens.reshape((-1,)),
            ignore_index=PAD_TOKEN,
            reduction="none",
        )

        return loss


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.LSTM(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.proj = nn.Linear(hidden_size * 2, vocab_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoded, tokens, h=None, c=None):
        x = self.embedding(tokens)

        if h is None or c is None:
            h = torch.zeros(
                (self.num_layers, x.shape[0], self.hidden_size), device=x.device
            )
            c = torch.zeros(
                (self.num_layers, x.shape[0], self.hidden_size), device=x.device
            )

        ## atencao
        outs = []
        for i in range(tokens.shape[1]):
            current_input = x[:, i : i + 1, :]  # [B, 1, EMBED]
            current_h = h[-1]  # [B, H]
            current_h = self.norm(current_h)

            # score = encoder out @ current h = [B, L, H] @ [B, H, 1] -> [B, L, 1]
            score = torch.bmm(encoded, current_h.unsqueeze(2)).squeeze(2)  # [B, L]
            soft_score = F.softmax(score, dim=-1).unsqueeze(1)  # [B, 1, L]
            context = torch.bmm(
                soft_score, encoded
            )  # [B, 1, L] @ [B, L, H] -> [B, 1, H]

            # concatear com o input
            input_decoder = torch.cat(
                [current_input, context], dim=2
            )  # [B, 1, EMBED + H]

            out, (h, c) = self.rnn(input_decoder, (h, c))
            # da para aproveitar e usar o contexto na projecao final tambem
            out_concat = torch.concat([out, context], dim=2)  # [B, 1, H + H]
            current_logits = self.proj(out_concat)
            outs.append(current_logits)

        outputs = torch.concat(outs, dim=1)
        return outputs, (h, c)


class AttentionCorrector(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_encoder,
        hidden_decoder,
        num_layers_encoder,
        num_layers_decoder,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            hidden_encoder,
            num_layers=num_layers_encoder,
            dropout=dropout,
        )

        self.decoder = AttentionDecoder(
            vocab_size,
            embed_dim,
            hidden_decoder,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )

        self.proj_encoded = nn.Linear(2 * hidden_encoder, hidden_decoder)
        self.norm_encoded = nn.LayerNorm(hidden_decoder)

    def forward(self, x_enc, x_dec, h_dec=None, c_dec=None):
        encoder_outputs, (h, c) = self.encoder(x_enc)
        # shape output = [B, L, 2 * hidden encoder]
        # precisa ser = [B, L, hidden decoder]
        encoded = self.proj_encoded(encoder_outputs)
        encoded = self.norm_encoded(encoded)

        logits, (h, c) = self.decoder(encoded, x_dec, h=h_dec, c=c_dec)
        return logits, (h, c)

    def train_step(self, batch):
        src_tokens, dec_tokens, trg_tokens = batch

        logits, (h, c) = self.forward(src_tokens, dec_tokens)
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            trg_tokens.reshape((-1,)),
            ignore_index=PAD_TOKEN,
        )

        return loss

    def valid_step(self, batch):
        src_tokens, dec_tokens, trg_tokens = batch

        logits, (h, c) = self.forward(src_tokens, dec_tokens)
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            trg_tokens.reshape((-1,)),
            ignore_index=PAD_TOKEN,
            reduction="none",
        )

        return loss


if __name__ == "__main__":
    corretor = AttentionCorrector(
        vocab_size=100,
        embed_dim=128,
        hidden_encoder=256,
        hidden_decoder=256,
        num_layers_encoder=2,
        num_layers_decoder=2,
    )
    corretor.to("cuda")

    x_src = torch.randint(0, 99, (2, 50), device="cuda")
    x_target = torch.randint(0, 99, (2, 50), device="cuda")
    with torch.no_grad():
        logits, (h, c) = corretor(x_src, x_target)
        print(logits.shape)
        print(h.shape)
        print(c.shape)
