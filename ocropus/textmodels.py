import torch
from torch import nn
from torchmore import combos, flex, layers
from typing import List, Tuple, Union
import scipy.ndimage as ndi
import numpy as np
from numpy import amax, arange, newaxis, tile
import sys
import typer
from . import ocrlayers, utils, transformer_utils
import editdistance

ninput = 3

app = typer.Typer()



ctc_loss = nn.CTCLoss(zero_infinity=True)

cross_entropy_loss = nn.CrossEntropyLoss(
    ignore_index= 261 #embedding_size (264) - 3
)  


def pack_for_ctc(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a list of tensors into tensors in the format required by CTC.

    Args:
        seqs (List[torch.Tensor]): list of tensors (integer sequences)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: packed tensor
    """
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def compute_ctc_loss(
    outputs: torch.Tensor, targets: List[torch.Tensor]
) -> torch.Tensor:
    global ctc_loss
    assert len(targets) == len(outputs)
    targets, tlens = pack_for_ctc(targets)
    b, d, L = outputs.size()
    olens = torch.full((b,), L, dtype=torch.long)
    outputs = outputs.log_softmax(1)
    outputs = layers.reorder(outputs, "BDL", "LBD")
    assert tlens.size(0) == b
    assert tlens.sum() == targets.size(0)
    return ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())


def ctc_decode(
    probs: torch.Tensor, sigma: float = 1.0, threshold: float = 0.7, full: bool = False
) -> Union[List[int], Tuple[List[int], List[float]]]:
    """Perform simple CTC decoding of the probability outputs from an LSTM or similar model.

    Args:
        probs (torch.Tensor): probabilities in BDL format.
        sigma (float, optional): smoothing. Defaults to 1.0.
        threshold (float, optional): thresholding of output probabilities. Defaults to 0.7.
        full (bool, optional): return both classes and probabilities. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], List[float]]]: [description]
    """
    if not isinstance(probs, np.ndarray):
        probs = probs.detach().cpu().numpy()
    probs = probs.T
    delta = np.amax(abs(probs.sum(1) - 1))
    assert delta < 1e-4, f"input not normalized ({delta}); did you apply .softmax()?"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, newaxis]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = tile(labels[:, newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


def charset_ascii():
    return "".join([chr(c) for c in range(128)])


class TextModel(nn.Module):
    """Word-level text model."""

    def __init__(
        self,
        mname,
        *,
        config={},
        charset: str = "ocropus.textmodels.charset_ascii",
        unknown_char: int = 26,
        no_ctc: bool = False,
        image_max_width = 2048
    ):
        super().__init__()
        self.charset = utils.load_symbol(charset)()
        self.no_ctc = no_ctc
        self.mname = mname
        self.noutput = len(self.charset)
        self.unknown_char = unknown_char
        self.model_uses_transformer = False
        self.image_max_width = image_max_width
        self.config = config

        self.init_model()

    def init_model(self):
        factory = utils.load_symbol(self.mname, default_module="ocropus.textmodels")
        self.model = factory(noutput=self.noutput, **self.config)


    @torch.jit.export
    def is_transformer_model(self):
        return self.model_uses_transformer

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        result = torch.zeros(len(s), dtype=torch.int64)
        for i, c in enumerate(s):
            result[i] = (
                self.charset.index(c) if c in self.charset else self.unknown_char
            )
        return result

    @torch.jit.export
    def decode_str(self, l: torch.Tensor) -> str:
        result = ""
        for c in l:
            result += (
                self.charset[c] if c < len(self.charset) else chr(self.unknown_char)
            )
        return result

    @torch.jit.export
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.min() >= 0.0 and images.max() <= 1.0
        b, c, h, w = images.shape
        for i in range(b):
            images[i] -= images[i].min()
            images[i] /= images[i].max() + 1e-6
        assert b >= 1 and b <= 16384
        assert c == 3
        assert h >= 12 and h <= 512 and w > 15 and w <= self.image_max_width 
        result = self.model.forward(images)
        assert result.shape[:2] == (b, len(self.charset))
        # assert result.shape[2] >= w - 32 and result.shape[2] <= w + 16, (images.shape, result.shape)
        return result

    def compute_loss(
        self, outputs: torch.Tensor, targets: List[torch.Tensor]
    ) -> torch.Tensor:
        if self.mname.startswith("ttext_"):
            raise NotImplementedError("transformer loss not implemented yet")
        else:
            return compute_ctc_loss(outputs, targets)

    @torch.jit.export
    def standardize(self, images: torch.Tensor) -> None:
        b, c, h, w = images.shape
        assert c == torch.tensor(1) or c == torch.tensor(3)
        if c == torch.tensor(1):
            images = images.repeat(1, 3, 1, 1)
            b, c, h, w = images.shape
        assert images.min() >= 0.0 and images.max() <= 1.0
        for i in range(len(images)):
            images[i] -= images[i].min()
            images[i] /= torch.max(
                images[i].amax(), torch.tensor([0.01], device=images[i].device)
            )
            if images[i].mean() > 0.5:
                images[i] = 1 - images[i]


class TransformerTextModel(TextModel):
    """Word-level text model."""

    def __init__(
        self,
        mname,
        *,
        config={},
        charset: str = "ocropus.textmodels.charset_ascii",
        unknown_char: int = 26,
        no_ctc: bool = True,
        transf_max_output_length = 32, #TODO: increase for text lines
        embedding_size = 264,
        max_input_sequence_length = 1024

    ):
        self.embedding_size = embedding_size
        self.max_input_sequence_length = max_input_sequence_length
        self.padding_pos = self.embedding_size - 3
        self.sos_pos = self.embedding_size - 2
        self.eos_pos = self.embedding_size - 1
        self.transf_max_output_length = transf_max_output_length 
        super().__init__(
            mname,
            config=config,
            charset=charset,
            unknown_char=unknown_char,
            no_ctc=no_ctc,
        )
        self.model_uses_transformer = True



    def init_model(self):
        factory = utils.load_symbol(self.mname, default_module="ocropus.textmodels")
        self.model = factory(noutput=self.noutput, last=self.embedding_size-4, embedding_size=self.embedding_size, max_input_sequence_length=self.max_input_sequence_length, **self.config)


    @torch.jit.export
    def is_transformer_model(self):
        return self.model_uses_transformer

    @torch.jit.export
    def encode_str(self, s: str) -> torch.Tensor:
        result = torch.zeros(len(s), dtype=torch.int64)
        for i, c in enumerate(s):
            result[i] = (
                self.charset.index(c) if c in self.charset else self.unknown_char
            )
        return result

    @torch.jit.export
    def decode_transf_string(self, l: torch.Tensor, include_eos: bool = True) -> str:
        extended_charset = {i: v for i, v in enumerate(self.charset)}
        # default_length = len(self.charset)
        extended_charset[self.padding_pos] = "+"
        extended_charset[self.sos_pos] = "<SOS>"
        extended_charset[self.eos_pos] = "<EOS>"
        result = ""
        for c in l:
            if c not in extended_charset:
                next_char = "<?>"
            else:
                next_char = extended_charset[c]
            if c == self.eos_pos:
                if include_eos is True:
                    result += next_char
                break
            else:
                result += next_char
        return result

    @torch.jit.export
    def strip_eos_and_padding_indices(self, token_indices: torch.Tensor) -> torch.Tensor:

        result = []
        #eos_pos_tensor = torch.tensor(self.eos_pos, dtype=torch.int64)
        end_indices = (token_indices == self.eos_pos).nonzero()
        if len(end_indices) == 0:
            return token_indices
        else:
            return token_indices[:end_indices[0][0]]

    @torch.jit.export
    def forward(
        self, images: torch.Tensor, encoded_text_targets: torch.Tensor
    ) -> torch.Tensor:
        assert images.min() >= 0.0 and images.max() <= 1.0
        b, c, h, w = images.shape
        for i in range(b):
            images[i] -= images[i].min()
            images[i] /= images[i].max() + 1e-6
        assert b >= 1 and b <= 16384
        assert c == 3
        assert h >= 12 and h <= 512 and w > 15 and w <= self.image_max_width 
        result = self.model.forward(images, encoded_text_targets)
        return result

    def compute_loss(
        self, outputs: torch.Tensor, encoded_targets_as_list: List[torch.Tensor]
    ) -> torch.Tensor:
        global cross_entropy_loss
        encoded_targets = encoded_targets_as_list[0]
        assert len(encoded_targets) == len(outputs)
        # reshape outputs and targets to (N,C,onehot)
        # flatten targets and outputs
        outputs = outputs.reshape(-1, outputs.size(2))
        encoded_targets = encoded_targets.reshape(-1).squeeze().cuda()
        assert outputs.size(0) == encoded_targets.size(0)
        loss = cross_entropy_loss(outputs, encoded_targets)
        return loss

    def add_sos_eos_and_padding(self, target_tokens, skip_sos=False, skip_eos=False):
        modified_targets = []
        for target_token in target_tokens:
            sos_token = torch.tensor([self.sos_pos])
            eos_token = torch.tensor([self.eos_pos])
            if len(target_token) > self.transf_max_output_length - 2:
                print(
                    "WARNING: target token ({}) is too long, truncating to {}".format(
                        target_token, self.transf_max_output_length - 2
                    )
                )
                target_token = target_token[: self.transf_max_output_length - 2]
            padding_length = self.transf_max_output_length - len(target_token) - 2
            new_list = [target_token]
            if skip_sos is True:
                padding_length += 1
            else:
                new_list = [sos_token] + new_list
            if skip_eos is True:
                padding_length += 1
            else:
                new_list = new_list + [eos_token]
            padding_token = torch.tensor([self.padding_pos] * padding_length)
            new_list += [padding_token]
            new_target_token = torch.concat(new_list, dim=0)
            modified_targets.append(new_target_token)
        return modified_targets

    def convert_target_batch_to_tensor(self, encoded_targets):
        y_concat = torch.stack(encoded_targets).permute(1, 0)
        return y_concat

    def compute_error(
        self,
        outputs: torch.Tensor,
        encoded_targets: List[torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_editdistance_error(outputs, encoded_targets)

    def compute_editdistance_error(
        self, outputs: torch.Tensor, encoded_targets_in_list: List[torch.Tensor]
    ) -> torch.Tensor:
        encoded_targets = encoded_targets_in_list[0]
        outputs_stripped_list = []
        targets_stripped_list = []

        encoded_targets = encoded_targets.transpose(0, 1).cpu()
        outputs_argmaxed = torch.argmax(outputs, dim=2).transpose(0, 1).cpu()
        assert outputs_argmaxed.shape == encoded_targets.shape

        for output in outputs_argmaxed:
            output_stripped = self.strip_eos_and_padding_indices(output)
            outputs_stripped_list.append(output_stripped.numpy().tolist())
        for target in encoded_targets:
            target = target
            target_stripped = self.strip_eos_and_padding_indices(target)
            targets_stripped_list.append(target_stripped.numpy().tolist())
        total = sum(len(t) for t in targets_stripped_list)
        errs = [
            editdistance.distance(p, t)
            for p, t in zip(outputs_stripped_list, targets_stripped_list)
        ]
        return sum(errs) / float(total)


@utils.model
def ctext_model_211124(noutput=None, shape=(1, ninput, 48, 300)):
    model = nn.Sequential(
        layers.ModPadded(
            64,
            combos.make_unet(
                [96, 96, 96, 128, 192, 256],
                sub=nn.Sequential(*combos.conv2d_block(256, 3, repeat=1)),
            ),
        ),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(128, 11, padding=11 // 2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_210910(noutput=None, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(100, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211217(noutput=None, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(100, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(300, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211215(
    noutput=None,
    shape=(1, ninput, 48, 300),
    depth=4,
    width=32,
    growth=1.5,
    xshrink=10,
    yshrink=10,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    fmpx = 1.0 / (float(xshrink) ** (1.0 / depth))
    fmpy = 1.0 / (float(yshrink) ** (1.0 / depth))
    initial = []
    if lstm_initial > 0:
        initial += flex.Lstm(lstm_initial)
    for i in range(depth):
        initial += combos.conv2d_block(
            int(width * (growth**depth)), fmp=(fmpy, fmpx), repeat=2
        )
    model = nn.Sequential(
        layers.KeepSize(sub=nn.Sequential(*initial)),
        flex.Lstm2(lstm_2d),
        ocrlayers.MaxReduce(2),
        # flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(width * 4, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(lstm_final, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211221(
    noutput=None,
    shape=(1, ninput, 64, 300),
    levels=4,
    depth=32,
    height=64,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [int(0.5 + 32 * (1.5**i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2**depth,
            combos.make_unet(depths, sub=flex.Lstm2d(depths[-1])),
        ),
        flex.Lstm2d(lstm_2d),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(depth * 4, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Lstm1d(lstm_final, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def ctext_model_211221(
    noutput=None,
    shape=(1, ninput, 48, 300),
    levels=4,
    depth=5,
    height=64,
    last=256,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model using only convolutions."""
    depths = [int(0.5 + 32 * (1.5**i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2**depth,
            combos.make_unet(depths, sub=flex.Conv2d(depths[-1], 3, padding=1)),
        ),
        flex.Conv2d(depth * 4, 3, padding=1),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def local_text_model_211221(
    noutput=None,
    shape=(1, ninput, 64, 300),
    levels=4,
    depth=5,
    height=64,
    last=256,
    growth=1.414,
    lstm_initial=0,
    lstm_2d=100,
    lstm_final=300,
):
    """Text recognition model with localization."""
    depths = [int(0.5 + 32 * (1.5**i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2**depth,
            combos.make_unet(depths, sub=flex.Conv2d(depths[-1], 3, padding=1)),
        ),
        flex.Conv2d(depth * 4, 3, padding=1),
        flex.BatchNorm2d(),
        nn.ReLU(),
        flex.Conv2d(noutput, 1),
        nn.Softmax(1),
        ocrlayers.MaxReduce(2),
        ocrlayers.Log(),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_211222(noutput=None, height=48, shape=(1, ninput, 48, 300), dropout=0.0):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [32, 64, 96, 128]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            16,
            combos.make_unet(depths, sub=flex.Lstm2d(196), dropout=dropout),
        ),
        flex.Lstm2(100),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(300, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        flex.Lstm1d(300, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def ctext_model_220117(
    noutput=None, height=48, lsize=128, shape=(1, ninput, 48, 300), dropout=0.0
):
    """Text recognition model using only convolutions."""
    depths = [lsize * s for s in [1, 2, 3, 4, 6]]
    fdepth = max(512, noutput // 2, depths[-1] * 2)
    print(f"# depths {depths}", file=sys.stderr)
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            32,
            combos.make_unet(
                depths, sub=flex.Conv2d(depths[-1], 3, padding=1), dropout=dropout
            ),
        ),
        *combos.conv2d_block(noutput // 4, 3, repeat=2),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(fdepth, 1, stride=2, padding=1),
        flex.Conv1d(fdepth, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        flex.Conv1d(fdepth, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@utils.model
def text_model_220204(
    noutput=None, height=48, shape=(1, ninput, 48, 300), complexity=1.0, dropout=0.5
):
    """Text recognition model using 2D LSTM and convolutions."""
    depths = [32, 64, 96, 128]
    depths = [int(x * complexity) for x in depths]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            16,
            combos.make_unet(depths, sub=flex.Lstm2d(196), dropout=dropout),
        ),
        flex.Lstm2(100),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        # layers.Fun("lambda x: x.max(2)[0]"),
        ocrlayers.MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2, padding=1),
        flex.Conv1d(300, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        *([nn.Dropout(dropout)] if dropout > 0.0 else []),
        flex.Lstm1d(300, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


@utils.model
def ttext_model_dev_1(
    noutput=None,
    shape=(1, ninput, 48, 300),
    levels=4,
    depth=5,
    height=64,
    last=260,
    embedding_size=264,
    max_input_sequence_length=1024,
):
    depths = [int(0.5 + 32 * (1.5**i)) for i in range(depth)]
    model = nn.Sequential(
        ocrlayers.HeightTo(height),
        layers.ModPadded(
            2**depth,
            combos.make_unet(depths, sub=flex.Conv2d(depths[-1], 3, padding=1)),
        ),
        flex.Conv2d(depth * 4, 3, padding=1),
        ocrlayers.MaxReduce(2),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(last, 3, padding=1),
        flex.BatchNorm1d(),
        nn.ReLU(),
    )
    flex.shape_inference(model, shape)

    padding_pos = embedding_size - 3
    assert last == embedding_size - 4

    class CombinedModel(nn.Module):
        def __init__(self, model, dim_cnn_out, padding_pos, num_classes):
            super(CombinedModel, self).__init__()
            self.conv_base_model = model
            self.dim_cnn_out = dim_cnn_out
            self.padded_decoder_input_dim = embedding_size
            assert dim_cnn_out == embedding_size-4
            self.padded_input_length = max_input_sequence_length 
            self.max_output_length = 32
            self.padding_pos = padding_pos

            # TODO: batch targets dynamically, based on current batch
            self.transformer_decoder = transformer_utils.TrDecoder(
                dim_input=dim_cnn_out + 4,
                pos_encoding_max_length= max_input_sequence_length,
                num_classes=num_classes,
            )

        def forward(self, x, encoded_text_targets):
            x = self.conv_base_model(x)
            batch_size = x.size(0)
            padded_cnn_output = torch.zeros(
                (batch_size, self.padded_decoder_input_dim, self.padded_input_length)
            ).cuda()
            padded_cnn_output[:, : x.size(1), : x.size(2)] = x
            (
                tgt_mask,
                memory_mask,
                tgt_padding_mask,
                memory_padding_mask,
            ) = transformer_utils.create_masks(
                encoded_text_targets,
                padded_cnn_output,
                self.padded_input_length,
                x.size(2),
                padding_index=self.padding_pos,
            )

            transformer_output = self.transformer_decoder(
                memory=padded_cnn_output,
                tgt=encoded_text_targets,
                tgt_mask=tgt_mask.cuda(),
                memory_mask=memory_mask.cuda(),
                tgt_key_padding_mask=tgt_padding_mask.cuda(),
                memory_key_padding_mask=memory_padding_mask.cuda(),
            )

            return transformer_output

    full_model = CombinedModel(model, dim_cnn_out=last, padding_pos=padding_pos, num_classes=embedding_size)
    return full_model


@utils.model
def text_model_small(noutput=None, shape=(1, ninput, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        flex.Conv2d(64, 3, padding=1),
        nn.ReLU(),
        flex.Lstm2(100),
        ocrlayers.MaxReduce(2),
        flex.Lstm1d(300, bidirectional=True),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


@app.command()
def list():
    for name, model in sorted(globals().items()):
        if "model" in name and callable(model):
            print(name)


@app.command()
def show(name: str):
    model = globals()[name]
    if not callable(model):
        raise ValueError(f"{name} is not a model")
    model = model(noutput=128)
    print(model)


if __name__ == "__main__":
    app()
