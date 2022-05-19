# -*- coding: utf-8 -*
"""private jit api for wenxin
"""
from paddle.fluid.dygraph import TracedLayer, Layer
from paddle.fluid.dygraph.base import program_desc_tracing_guard
from paddle.fluid.dygraph.jit import create_program_from_desc
from paddle.fluid.framework import dygraph_only
from paddle.fluid.framework import _dygraph_guard, _dygraph_tracer


class WenxinTracedLayer(TracedLayer):
    """WenxinTracedLayer
    """
    def __init__(self, program, parameters, feed_names, fetch_names):
        """
        :param program
        :param parameters
        :param feed_names
        :param fetch_names
        """
        TracedLayer.__init__(self, program, parameters, feed_names, fetch_names)

    @staticmethod
    @dygraph_only
    def trace(layer, inputs, phase="save_inference"):
        """
        :param layer (paddle.nn.Layer): the layer object to be traced.
        :param inputs (list(Tensor)|tuple(Tensor)|Tensor): the input tensors of the layer object.
        :param phase
        :return A tuple of 2 items, whose the first item is the output of code:`layer(*inputs)` ,
        and the second item is the created TracedLayer object.
        """
        assert isinstance(
            layer, Layer
        ), "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received {}."\
            .format(type(layer))
        out = _trace_wenxin(layer, inputs, phase="save_inference")
        outs = out[0]
        prog = out[1]
        feed = out[2]
        fetch = out[3]
        parameters = out[4]

        traced = TracedLayer(prog, parameters, feed, fetch)
        return outs, traced



@dygraph_only
def _trace_wenxin(layer,
                  inputs,
                  phase="save_inference",
                  feed_prefix='feed_',
                  fetch_prefix='fetch_',
                  tmp_prefix='t_'):
    """
    :param layer
    :param inputs
    :param phase
    :param feed_prefix
    :param fetch_prefix
    :param tmp_prefix
    """
    assert isinstance(layer, Layer)

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    with program_desc_tracing_guard(True):
        original_outputs = layer(inputs, phase)

        var_list = original_outputs.get("target_feed")
        out_vars = original_outputs.get("target_predicts")

        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(var_list, feed_prefix, out_vars,
                                                                                       fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return [original_outputs, program, feed_names, fetch_names, parameters]




