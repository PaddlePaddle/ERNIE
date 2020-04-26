from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import argparse
import json
import logging
import logging
from functools import partial

import paddle.fluid.dygraph as D
import paddle.fluid as F
import paddle.fluid.layers as L

from ernie.file_utils import _fetch_from_remote

log = logging.getLogger(__name__)

def _build_linear(n_in, n_out, name, init, act=None):
    return D.Linear(n_in, 
            n_out, 
            param_attr=F.ParamAttr(name='%s.w_0' % name if name is not None else None, initializer=init), 
            bias_attr='%s.b_0' % name if name is not None else None, act=act)


def _build_ln(n_in, name):
    return D.LayerNorm(normalized_shape=n_in, 
            param_attr=F.ParamAttr(name='%s_layer_norm_scale' % name if name is not None else None, 
                initializer=F.initializer.Constant(1.)),
            bias_attr=F.ParamAttr(name='%s_layer_norm_bias' % name if name is not None else None, 
                initializer=F.initializer.Constant(1.)),
        )


def append_name(name, postfix):
    if name is None:
        return None
    elif name == '':
        return postfix
    else:
        return '%s_%s' % (name, postfix)


class AttentionLayer(D.Layer):
    def __init__(self, cfg, name=None):
        super(AttentionLayer, self).__init__()
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head', d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head', d_model // n_head) * n_head
        self.n_head = n_head
        self.d_key = d_model_q // n_head
        self.q = _build_linear(d_model, d_model_q, append_name(name, 'query_fc'), initializer)
        self.k = _build_linear(d_model, d_model_q, append_name(name, 'key_fc'), initializer)
        self.v = _build_linear(d_model, d_model_v, append_name(name, 'value_fc'), initializer)
        self.o = _build_linear(d_model_v, d_model, append_name(name, 'output_fc'), initializer)
        self.dropout = lambda i: L.dropout(i, dropout_prob=cfg['attention_probs_dropout_prob'], dropout_implementation="upscale_in_train",) if self.training else i

    def forward(self, queries, keys, values, attn_bias):
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        #bsz, q_len, q_dim = queries.shape
        #bsz, k_len, k_dim = keys.shape
        #bsz, v_len, v_dim = values.shape
        #assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        q = L.transpose(L.reshape(q, [0, 0, self.n_head, q.shape[-1] // self.n_head]), [0, 2, 1, 3]) #[batch, head, seq, dim]
        k = L.transpose(L.reshape(k, [0, 0, self.n_head, k.shape[-1] // self.n_head]), [0, 2, 1, 3]) #[batch, head, seq, dim]
        v = L.transpose(L.reshape(v, [0, 0, self.n_head, v.shape[-1] // self.n_head]), [0, 2, 1, 3]) #[batch, head, seq, dim]

        q = L.scale(q, scale=self.d_key ** -0.5)
        score = L.matmul(q, k, transpose_y=True)
        if attn_bias is not None:
            score += attn_bias
        score = L.softmax(score, use_cudnn=True)
        score = self.dropout(score)

        out = L.matmul(score, v)
        out = L.transpose(out, [0, 2, 1, 3])
        out = L.reshape(out, [0, 0, out.shape[2] * out.shape[3]])

        out = self.o(out)
        return out


class PositionwiseFeedForwardLayer(D.Layer):
    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardLayer, self).__init__()
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        assert cfg['hidden_act'] in ['relu', 'gelu']
        self.i = _build_linear(d_model, d_ffn, append_name(name, 'fc_0'), initializer, act=cfg['hidden_act'])
        self.o = _build_linear(d_ffn, d_model, append_name(name, 'fc_1'), initializer)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    def forward(self, inputs):
        hidden = self.i(inputs)
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


class ErnieBlock(D.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieBlock, self).__init__()
        d_model = cfg['hidden_size']
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])

        self.attn = AttentionLayer(cfg, name = append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name = append_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(cfg, name = append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name = append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    def forward(self, inputs, attn_bias=None):
        attn_out = self.attn(inputs, inputs, inputs, attn_bias) #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs 
        hidden = self.ln1(hidden) # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden

        
class ErnieEncoderStack(D.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = D.LayerList([ErnieBlock(cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers)])

    def forward(self, inputs, attn_bias=None):
        for b in self.block:
            inputs = b(inputs, attn_bias)
        return inputs


class PretrainedModel(object):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {
        'ernie-1.0': bce + 'model-ernie1.0.1.tar.gz',
        'ernie-2.0-en': bce + 'model-ernie2.0-en.1.tar.gz',
        'ernie-2.0-large-en':  bce + 'model-ernie2.0-large-en.1.tar.gz',
        'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz',
    }
    @classmethod
    def from_pretrained(cls, pretrain_dir_or_url, force_download=False, **kwargs):
        if pretrain_dir_or_url in cls.resource_map:
            url = cls.resource_map[pretrain_dir_or_url]
            log.info('get pretrain dir from %s' % url)
            pretrain_dir = _fetch_from_remote(url, force_download)
        else:
            log.info('pretrain dir %s not in %s, read from local' % (pretrain_dir_or_url, repr(cls.resource_map)))
            pretrain_dir = pretrain_dir_or_url

        if not os.path.exists(pretrain_dir):
            raise ValueError('pretrain dir not found: %s' % pretrain_dir)
        param_path = os.path.join(pretrain_dir, 'params')
        state_dict_path = os.path.join(pretrain_dir, 'saved_weights')
        config_path = os.path.join(pretrain_dir, 'ernie_config.json')

        if not os.path.exists(config_path):
            raise ValueError('config path not found: %s' % config_path)
        name_prefix=kwargs.pop('name', None)
        cfg_dict = dict(json.loads(open(config_path).read()), **kwargs)
        model = cls(cfg_dict, name=name_prefix)
        
        log.info('loading pretrained model from %s' % pretrain_dir)

        #if os.path.exists(param_path):
        #    raise NotImplementedError()
        #    log.debug('load pretrained weight from program state')
        #    F.io.load_program_state(param_path) #buggy in dygraph.gurad, push paddle to fix
        if os.path.exists(state_dict_path + '.pdparams'):
            m, _ = D.load_dygraph(state_dict_path)
            for k, v in model.state_dict().items():
                if k not in m:
                    log.warn('param:%s not set in pretrained model, skip' % k)
                    m[k] = v # FIXME: no need to do this in the future
            model.set_dict(m)
        else:
            raise ValueError('weight file not found in pretrain dir: %s' % pretrain_dir)
        return model


class ErnieModel(D.Layer, PretrainedModel):
    def __init__(self, cfg, name=None):
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        D.Layer.__init__(self)
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.n_head = cfg['num_attention_heads']
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = D.Embedding([d_vocab, d_emb], param_attr=F.ParamAttr(name=append_name(name, 'word_embedding'), initializer=initializer))
        self.pos_emb = D.Embedding([d_pos, d_emb], param_attr=F.ParamAttr(name=append_name(name, 'pos_embedding'), initializer=initializer))
        self.sent_emb = D.Embedding([d_sent, d_emb], param_attr=F.ParamAttr(name=append_name(name, 'sent_embedding'), initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

        self.encoder_stack = ErnieEncoderStack(cfg, append_name(name, 'encoder'))
        self.pooler = _build_linear(cfg['hidden_size'], cfg['hidden_size'], append_name(name, 'pooled_fc'), initializer, act='tanh')
        self.train()
    

    def eval(self):
        if F.in_dygraph_mode():
            super(ErnieModel, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False

    def train(self):
        if F.in_dygraph_mode():
            super(ErnieModel, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True

    def forward(self, src_ids, sent_ids=None, pos_ids=None, input_mask=None, attn_bias=None):
        #d_batch, d_seqlen = src_ids.shape
        assert len(src_ids.shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (repr(src_ids.shape))
        d_batch = L.shape(src_ids)[0]
        d_seqlen = L.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = L.reshape(L.range(0, d_seqlen, 1, dtype='int32'), [1, -1])
            pos_ids = L.cast(pos_ids, 'int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = L.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = L.unsqueeze(input_mask, axes=[1,3])
            attn_bias = (1. - L.matmul(input_mask, input_mask, transpose_y=True)) * -10000.0
            attn_bias = L.expand(attn_bias, [1, self.n_head, 1, 1]) # avoid broadcast =_=
            attn_bias.stop_gradient = True
            
        if sent_ids is None:
            sent_ids = L.zeros_like(src_ids)

        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded

        embedded = self.dropout(self.ln(embedded))

        encoded = self.encoder_stack(embedded, attn_bias)
        pooled = self.pooler(encoded[:, 0, :])

        return pooled, encoded
        

class ErnieModelForSequenceClassification(ErnieModel):
    def __init__(self, cfg, name=None):
        super(ErnieModelForSequenceClassification, self).__init__(cfg, name=name)

        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        self.cls_fc = _build_linear(cfg['hidden_size'], cfg['num_labels'], append_name(name, 'cls'), initializer)

        prob = cfg.get('pooler_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    def forward(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        pooled, encoded = super(ErnieModelForSequenceClassification, self).forward(*args, **kwargs)
        hidden = self.dropout(pooled)
        logits = self.cls_fc(hidden)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


class ErnieModelForTokenClassification(ErnieModel):
    def __init__(self, cfg, name=None):
        super(ErnieModelForTokenClassification, self).__init__(cfg, name=name)

        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        self.cls_fc = _build_linear(cfg['hidden_size'], cfg['num_labels'], append_name(name, 'cls'), initializer)

        prob = cfg.get('pooler_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: L.dropout(i, dropout_prob=prob, dropout_implementation="upscale_in_train",) if self.training else i

    def forward(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        pooled, encoded = super(ErnieModelForTokenClassification, self).forward(*args, **kwargs)
        hidden = self.dropout(encoded) # maybe not?
        logits = self.cls_fc(hidden)

        if labels is not None:
            if len(labels.shape) == 2:
                labels = L.unsqueeze(labels, axes=[-1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


class NSPHead(D.Layer):
    def __init__(self, cfg, name=None):
        super(NSPHead, self).__init__()
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        self.nsp = _build_linear(cfg['hidden_size'], 2, append_name(name, 'nsp_fc'), initializer)

    def forward(self, inputs, labels):
        logits = self.nsp(inputs)
        loss = L.softmax_with_cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
        return loss


class ErnieModelForPretraining(ErnieModel):
    def __init__(self, cfg, name=None):
        super(ErnieModelForPretraining, self).__init__(cfg, name=name)
        initializer = F.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_vocab = cfg['vocab_size']

        self.pooler_heads = D.LayerList([NSPHead(cfg, name=name)])
        self.mlm = _build_linear(d_model, d_model, append_name(name, 'mask_lm_trans_fc'), initializer, act=cfg['hidden_act'])
        self.mlm_ln = _build_ln(d_model, name = append_name(name, 'mask_lm_trans'))
        self.mlm_bias = L.create_parameter(
                dtype='float32',
                shape=[d_vocab], 
                attr=F.ParamAttr(
                    name=append_name(name, 'mask_lm_out_fc.b_0'), 
                    initializer=F.initializer.Constant(value=0.0)
                    ),
                is_bias=True,
                )

    def forward(self, *args, **kwargs):
        mlm_labels = kwargs.pop('labels')
        mlm_pos = kwargs.pop('mlm_pos')
        nsp_labels = kwargs.pop('nsp_labels')
        pooled, encoded = super(ErnieModelForPretraining, self).forward(*args, **kwargs)
        if len(mlm_labels.shape) == 1:
            mlm_labels = L.reshape(mlm_labels, [-1, 1])
        if len(nsp_labels.shape) == 1:
            nsp_labels = L.reshape(nsp_labels, [-1, 1])

        nsp_loss = self.pooler_heads[0](pooled, nsp_labels)

        encoded_2d = L.gather_nd(encoded, mlm_pos)
        encoded_2d = self.mlm(encoded_2d)
        encoded_2d = self.mlm_ln(encoded_2d)
        logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
        mlm_loss = L.reduce_mean(L.softmax_with_cross_entropy(logits_2d, mlm_labels))
        total_loss = mlm_loss + nsp_loss
        return total_loss, mlm_loss, nsp_loss

