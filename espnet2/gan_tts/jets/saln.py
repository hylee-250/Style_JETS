# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging

import torch
from torch.nn import functional as F
import numpy as np

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)

class SALNEncoder(torch.nn.Module):
    """Transformer Style encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        conv_wshare (int): The number of kernel of convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]): Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in selfattention_layer_type
            == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): Whether to use bias in convolution. Only used in
            selfattention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        pos_enc_class (torch.nn.Module): Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        selfattention_layer_type (str): Encoder attention layer type.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(self,idim,attention_dim=256,attention_heads=4,conv_wshare=4,
                    conv_kernel_length="11",conv_usebias=False,linear_units=2048,
                    num_blocks=6,dropout_rate=0.1,
                    positional_dropout_rate=0.1,
                    attention_dropout_rate=0.0,input_layer="conv2d",
                    pos_enc_class=PositionalEncoding,normalize_before=True,
                    concat_after=False,positionwise_layer_type="conv1d",
                    positionwise_conv_kernel_size=1,selfattention_layer_type="selfattn",
                    padding_idx=-1,stochastic_depth_rate=0.0,intermediate_layers=None,
                    ctc_softmax=None,conditioning_layer_dim=None,style_dim=128,):
        """Construct an SALNEncoder object."""
        super(SALNEncoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        self.conv_subsampling_factor = 1

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
            self.conv_subsampling_factor = 8
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )

        if selfattention_layer_type in ["selfattn","rel_selfattn","legacy_rel_selfattn",]:
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = [
                (
                    attention_heads,
                    attention_dim,
                    attention_dropout_rate,
                )
            ] * num_blocks
            
        elif selfattention_layer_type == "lightconv":
            logging.info("encoder self-attention layer type = lightweight convolution")
            encoder_selfattn_layer = LightweightConvolution
            encoder_selfattn_layer_args = [
                (conv_wshare,attention_dim,attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),False,conv_usebias,)
                for lnum in range(num_blocks)
            ]
        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "encoder self-attention layer "
                "type = lightweight convolution 2-dimensional"
            )
            encoder_selfattn_layer = LightweightConvolution2D
            encoder_selfattn_layer_args = [
                (conv_wshare,attention_dim,attention_dropout_rate,
                    int(conv_kernel_length.split("_")[lnum]),False,conv_usebias,)
                for lnum in range(num_blocks)
            ]

        # FFT Block 6개
        # self.encoders = repeat(num_blocks,
        #     lambda lnum: SALNEncoderLayer(
        #         attention_dim,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
        #         positionwise_layer(*positionwise_layer_args),
        #         style_dim,
        #         dropout_rate,
        #         normalize_before,
        #         concat_after,
        #     ),
        # )

        self.encoders = torch.nn.ModuleList([SALNEncoderLayer(
                attention_dim,
                encoder_selfattn_layer(attention_heads,attention_dim,attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                style_dim,
                dropout_rate,
                normalize_before,
                concat_after,)
                for _ in range(num_blocks)])
            # self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            # self.fft_conv1d_kernel_size, self.style_dim, self.dropout) for _ in range(self.n_layers)])

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, attention_dim
            )

    def get_positionwise_layer(
        self,positionwise_layer_type="linear",attention_dim=256,
        linear_units=2048,dropout_rate=0.1,positionwise_conv_kernel_size=1,):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, style_vector,masks=None):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """

        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
 

            slf_attn = []
            for enc_layer in self.encoders:
                # def forward(self, x, style_vector, mask=None,self_attn_mask=None, cache=None):   
                xs, enc_slf_attn = enc_layer(
                    xs, style_vector, 
                    mask=masks, )
                slf_attn.append(enc_slf_attn)

        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, style_vector ,masks)

                if (self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers):
                    encoder_output = xs
                    # intermediate branches also require normalization.
                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)
                        xs = xs + self.conditioning_layer(intermediate_result)

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks


# FFTBlock과 같은 모듈
class SALNEncoderLayer(torch.nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension -> attention_dim
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(self,attention_dim,self_attn,feed_forward,style_dim,dropout_rate,
                    normalize_before=True,concat_after=False,
                    fft_conv1d_kernel_size= [9, 1],):
        """Construct an SALNEncoderLayer object."""
        super(SALNEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # self.feed_forward = feed_forward
        self.feed_forward = PositionwiseFeedForward(attention_dim, 1024,fft_conv1d_kernel_size,dropout=0.1)
#            def __init__(self, idim, hidden_units,fft_conv1d_kernel_size,dropout=0.1):

        self.norm1 = LayerNorm(attention_dim)
        self.norm2 = LayerNorm(attention_dim)

        # style vector dimension : 128
        self.saln_0 = StyleAdaptiveLayerNorm(attention_dim,style_dim)
        self.saln_1 = StyleAdaptiveLayerNorm(attention_dim,style_dim)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.attention_dim = attention_dim
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(attention_dim + attention_dim, attention_dim)

    def forward(self, x, style_vector, mask=None,self_attn_mask=None, cache=None):   
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        residual = x

        # print('------------x:',x)
        # self.normalize_before=true
        if self.normalize_before:
            x = self.norm1(x)
        # print('------------normalized x:',x)

        x_q = x
        x = residual + self.dropout(self.self_attn(x, x, x, mask))

        x = self.saln_0(x,style_vector)
  
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        x = self.saln_1(x,style_vector)

        return x, mask


class PositionwiseFeedForward(torch.nn.Module):
    ''' A two-feed-forward-layer module '''    
    
    def __init__(self, idim, hidden_units,fft_conv1d_kernel_size,dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(idim, hidden_units, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 = ConvNorm(hidden_units, idim, kernel_size=fft_conv1d_kernel_size[1])


        #  def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):        
        '''
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.encoder_layer
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.decoder_hidden
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout
        '''

        self.mish = Mish()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self).forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )

# SALN
class StyleAdaptiveLayerNorm(torch.nn.Module):
    def __init__(self, in_channel, style_dim=128):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = torch.nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style vector를 affine layer에 통과시켜서 감마와 베타를 얻는 과정
        
        style = self.style(style_code).unsqueeze(1)     # [42,128] -> AffineLinear(128,512) -> [42,512]
        # torch.chunk를 통해서 2배만큼 늘리고 두 개로 나누기
        gamma, beta = style.chunk(2, dim=-1)
        
        out = self.norm(input)
        out = gamma * out + beta
        
        return out

class MelStyleEncoder(torch.nn.Module):
    def __init__(self,n_mel = 80, hidden_dim=128, out_dim=128,kernel_size=5,
                    n_head=2,dropout=0.1):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout

        self.spectral = torch.nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),Mish(),
            torch.nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),Mish(),
            torch.nn.Dropout(self.dropout)
        )

        self.temporal = torch.nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, mel, mask=None):
        max_len = mel.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None

        # spectral
        mel = self.spectral(mel)

        # temporal
        mel = mel.transpose(1,2)
        mel = self.temporal(mel)
        mel = mel.transpose(1,2)
        # self-attention
        if mask is not None:
            mel = mel.masked_fill(mask.unsqueeze(-1), 0)
        mel, _ = self.slf_attn(mel, mask=slf_attn_mask)
        # fc
        mel = self.fc(mel)
        # temoral average pooling


        w = self.temporal_avg_pool(mel, mask=mask)

        return w

class AffineLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = torch.nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AffineLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = torch.nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)

class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        self.dropout = torch.nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = torch.nn.utils.spectral_norm(self.w_qs)
            self.w_ks = torch.nn.utils.spectral_norm(self.w_ks)
            self.w_vs = torch.nn.utils.spectral_norm(self.w_vs)
            self.fc = torch.nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class LinearNorm(torch.nn.Module):
    def __init__(self,in_channels,out_channels,bias=True, spectral_norm=False,):
        super(LinearNorm, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = torch.nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out

        
class ConvNorm(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,
                 padding=None,dilation=1,bias=True,spectral_norm=False,):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,out_channels,
                                    kernel_size=kernel_size,stride=stride,
                                    padding=padding,dilation=dilation,bias=bias)
        if spectral_norm:
            self.conv = torch.nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out



class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class Conv1dGLU(torch.nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
        self.dropout = torch.nn.Dropout(dropout)
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x