import paddle
from paddle import nn


class MLMLoss(nn.Layer):
    def __init__(self,
                 lsm_weight: float=0.1,
                 ignore_id: int=-1,
                 text_masking: bool=False):
        super().__init__()
        if text_masking:
            self.text_mlm_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
        if lsm_weight > 50:
            self.l1_loss_func = nn.MSELoss()
        else:
            self.l1_loss_func = nn.L1Loss(reduction='none')
        self.text_masking = text_masking

    def forward(self,
                speech: paddle.Tensor,
                before_outs: paddle.Tensor,
                after_outs: paddle.Tensor,
                masked_pos: paddle.Tensor,
                text: paddle.Tensor=None,
                text_outs: paddle.Tensor=None,
                text_masked_pos: paddle.Tensor=None):

        xs_pad = speech
        mlm_loss_pos = masked_pos > 0
        loss = paddle.sum(
            self.l1_loss_func(
                paddle.reshape(before_outs, (-1, self.odim)),
                paddle.reshape(xs_pad, (-1, self.odim))),
            axis=-1)
        if after_outs is not None:
            loss += paddle.sum(
                self.l1_loss_func(
                    paddle.reshape(after_outs, (-1, self.odim)),
                    paddle.reshape(xs_pad, (-1, self.odim))),
                axis=-1)
        loss_mlm = paddle.sum((loss * paddle.reshape(
            mlm_loss_pos, [-1]))) / paddle.sum((mlm_loss_pos) + 1e-10)

        if self.text_masking:
            loss_text = paddle.sum((self.text_mlm_loss(
                paddle.reshape(text_outs, (-1, self.vocab_size)),
                paddle.reshape(text, (-1))) * paddle.reshape(
                    text_masked_pos,
                    (-1)))) / paddle.sum((text_masked_pos) + 1e-10)

            return loss_mlm, loss_text

        return loss_mlm
