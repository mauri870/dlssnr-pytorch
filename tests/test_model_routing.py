import torch
from torch import nn

from dlssnr.model import NRNet
from dlssnr.plan import KINDS


class Recorder(nn.Module):
    """A layer that returns its own index (twice, as a tuple) and records what it was given."""

    def __init__(self, spec, log):
        super().__init__()
        self.spec, self.log = spec, log

    def forward(self, *args):
        self.log[self.spec.index] = args
        return (self.spec.index, -self.spec.index)


def _net():
    log: dict = {}
    builders = {kind: (lambda spec, weights: Recorder(spec, log)) for kind in KINDS.values()}
    return NRNet(None, 1920, 1080, builders=builders), log


def test_skip_sources_point_backwards_and_match_the_plan():
    net, _ = _net()
    index = {(s.block, s.layer): s.index for s in net.plan}
    for spec, source in zip(net.plan, net.skip_source):
        if source is not None:
            assert source < spec.index
    for up, down in ((48, 22), (56, 14), (62, 8), (66, 4), (70, 0)):
        assert net.skip_source[index[up, 0]] == index[down, 0]
    assert net.skip_source[index[39, 0]] == index[30, 3]


def test_every_layer_runs_in_order():
    net, log = _net()
    outputs = net.run(torch.zeros(1, 3, 1088, 1920), capture=True)
    assert sorted(log) == list(range(152)) and len(outputs) == 152
    index = {(s.block, s.layer): s.index for s in net.plan}
    assert log[index[66, 0]][0] == index[65, 0]
    assert log[index[66, 0]][1] == (index[4, 0], -index[4, 0])
    assert log[index[70, 0]][1] == -index[0, 0]
