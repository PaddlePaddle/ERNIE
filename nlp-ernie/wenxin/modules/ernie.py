# -*- coding: utf-8 -*
"""
ERNIE 网络结构
"""

import logging

import paddle
from paddle import nn
from paddle.nn import functional as F


ACT_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}

class ErnieModel(nn.Layer):
    """ ernie model """

    def __init__(self, cfg, name=''):
        """
        Fundamental pretrained Ernie model
        """
        nn.Layer.__init__(self)
        self.cfg = cfg
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
#         d_sent = cfg.get("sent_type_vocab_size", 4) or cfg.get('type_vocab_size', 4)
        if cfg.has('sent_type_vocab_size'):
            d_sent = cfg['sent_type_vocab_size']
        else:
            d_sent = cfg.get('type_vocab_size', 2)
            
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        self.initializer = nn.initializer.TruncatedNormal(std=cfg['initializer_range'])

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=paddle.ParamAttr(name=append_name(name, 'word_embedding'), initializer=self.initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=paddle.ParamAttr(name=append_name(name, 'pos_embedding'), initializer=self.initializer))
#         self.sent_emb = nn.Embedding(
#             d_sent,
#             d_emb,
#             weight_attr=paddle.ParamAttr(name=append_name(name, 'sent_embedding'), initializer=self.initializer))
        self._use_sent_id = cfg.get('use_sent_id', True)
        if self._use_sent_id:
            self.sent_emb = nn.Embedding(
                d_sent,
                d_emb,
                weight_attr=paddle.ParamAttr(name=append_name(name, 'sent_embedding'), initializer=self.initializer))
        self._use_task_id = cfg.get('use_task_id', False)
        if self._use_task_id:
            self._task_types = cfg.get('task_type_vocab_size', 3)
            logging.info('using task_id, #task_types:{}'.format(self._task_types))
            self.task_emb = nn.Embedding(
            self._task_types,
            d_emb,
            weight_attr=paddle.ParamAttr(name=append_name(name, 'task_embedding'), initializer=self.initializer))

        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)

        self.encoder_stack = ErnieEncoderStack(cfg, append_name(name, 'encoder'))

        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(cfg['hidden_size'], cfg['hidden_size'], append_name(name, 'pooled_fc'),
                self.initializer)
        else:
            self.pooler = None

        self.train()

    # FIXME:remove this
    def eval(self):
        """ eval """
        if paddle.in_dynamic_mode():
            super(ErnieModel, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        """ train """
        if paddle.in_dynamic_mode():
            super(ErnieModel, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

    def forward(self,
                src_ids,
                sent_ids=None,
                pos_ids=None,
                input_mask=None,
                task_ids=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        """
        Args:
            src_ids (`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (optional, `Variable` of shape `[batch_size, seq_len]`):
                aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(optional, `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(optional `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            task_ids(optional `Variable` of shape `[batch_size, seq_len]`):
                task type for pre_train task type
            attn_bias(optional, `Variable` of shape `[batch_size, seq_len, seq_len] or False`):
                3D version of `input_mask`, if set, overrides `input_mask`; if set not False, will not apply attention mask
            past_cache(optional, tuple of two lists: cached key and cached value,
                each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`):
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.

        Returns:
            pooled (`Variable` of shape `[batch_size, hidden_size]`):
                output logits of pooler classifier
            encoded(`Variable` of shape `[batch_size, seq_len, hidden_size]`):
                output logits of transformer stack
            info (Dictionary):
                addtional middle level info, inclues: all hidden stats, k/v caches.
        """
        assert len(src_ids.shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` specified; attn_bias must not be None'
        d_seqlen = paddle.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = paddle.arange(0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')

        if attn_bias is None:
            if input_mask is None:
                input_mask = paddle.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = paddle.reshape(paddle.arange(0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(attn_bias.shape) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape

        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile([1, self.n_head, 1, 1])   # avoid broadcast =_=

        if sent_ids is None:
            sent_ids = paddle.zeros_like(src_ids)

        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
#         sent_embedded = self.sent_emb(sent_ids)
#         embedded = src_embedded + pos_embedded + sent_embedded
        embedded = src_embedded + pos_embedded
        if self._use_sent_id:
            sent_embedded = self.sent_emb(sent_ids)
            embedded = embedded + sent_embedded
        if self._use_task_id:
            task_embeded = self.task_emb(task_ids)
            embedded = embedded + task_embeded

        embedded = self.dropout(self.ln(embedded))

        encoded, hidden_list, cache_list = self.encoder_stack(embedded, attn_bias, past_cache=past_cache)

        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None

        additional_info = {
            'hiddens': hidden_list,
            'caches': cache_list,
        }

        if self.return_additional_info:
            return pooled, encoded, additional_info
        return pooled, encoded


class ErnieEncoderStack(nn.Layer):
    """ ernie encoder stack """

    def __init__(self, cfg, name=None):
        super(ErnieEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.LayerList([
            ErnieBlock(cfg, append_name(name, 'layer_%d' % i))
            for i in range(n_layers)
        ])

    def forward(self, inputs, attn_bias=None, past_cache=None):
        """ forward function """
        if past_cache is not None:
            assert isinstance(
                past_cache, tuple
            ), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs, attn_bias=attn_bias, past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)


class ErnieBlock(nn.Layer):
    """ ernie block class """

    def __init__(self, cfg, name=None):
        super(ErnieBlock, self).__init__()
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(cfg, name=append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'))
        self.ffn = PositionWiseFeedForwardLayer(cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs, attn_bias=None, past_cache=None):
        """ forward """
        attn_out, cache = self.attn(inputs, inputs, inputs, attn_bias, past_cache=past_cache)  # self attention
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class AttentionLayer(nn.Layer):
    """ attention layer """

    def __init__(self, cfg, name=None):
        super(AttentionLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(std=cfg['initializer_range'])
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
        self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

    def forward(self, queries, keys, values, attn_bias, past_cache):
        """ layer forward function """
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        # bsz, q_len, q_dim = queries.shape
        # bsz, k_len, k_dim = keys.shape
        # bsz, v_len, v_dim = values.shape
        # assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = paddle.concat([cached_k, k], 1)
            v = paddle.concat([cached_v, v], 1)

        # [batch, head, seq, dim]
        q = q.reshape([0, 0, self.n_head, q.shape[-1] // self.n_head]).transpose([0, 2, 1, 3])
        # [batch, head, seq, dim]
        k = k.reshape([0, 0, self.n_head, k.shape[-1] // self.n_head]).transpose([0, 2, 1, 3])
        # [batch, head, seq, dim]
        v = v.reshape([0, 0, self.n_head, v.shape[-1] // self.n_head]).transpose([0, 2, 1, 3])
        q = q.scale(self.d_key ** -0.5)
        score = q.matmul(k, transpose_y=True)

        if attn_bias is not None:
            score += attn_bias
        score = F.softmax(score)
        score = self.dropout(score)

        out = score.matmul(v).transpose([0, 2, 1, 3])
        out = out.reshape([0, 0, out.shape[2] * out.shape[3]])
        out = self.o(out)
        return out, cache


class PositionWiseFeedForwardLayer(nn.Layer):
    """ post wise feed forward layer """

    def __init__(self, cfg, name=None):
        super(PositionWiseFeedForwardLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)

        self.act = ACT_DICT[cfg['hidden_act']]()
        self.i = _build_linear(d_model, d_ffn, append_name(name, 'fc_0'), initializer)
        self.o = _build_linear(d_ffn, d_model, append_name(name, 'fc_1'), initializer)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        """ forward """
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


def _build_linear(n_in, n_out, name, init):
    """
    """
    return nn.Linear(
        n_in,
        n_out,
        weight_attr=paddle.ParamAttr(
            name='%s.w_0' % name if name is not None else None,
            initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None)


def _build_ln(n_in, name):
    """
    """
    return nn.LayerNorm(
        normalized_shape=n_in,
        weight_attr=paddle.ParamAttr(
            name='%s_layer_norm_scale' % name if name is not None else None,
            initializer=nn.initializer.Constant(1.)),
        bias_attr=paddle.ParamAttr(
            name='%s_layer_norm_bias' % name if name is not None else None,
            initializer=nn.initializer.Constant(0.)))


def append_name(name, postfix):
    """ append name with postfix """
    if name is None:
        ret = None
    elif name == '':
        ret = postfix
    else:
        ret = '%s_%s' % (name, postfix)
    return ret


