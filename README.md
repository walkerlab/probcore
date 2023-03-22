# GenSn (gèńséń)
A library for generative and sampling-based modeling

## What does GenSn stand for?
The library name GenSn (pronounced gen-sen with hard G as in get) stands for the Japanese word gensen (源泉) which can loosely be translated as source, espeically as source of a stream (of water).

GenSn also stands for **gen**erative **sn**(samples, as in $S_n$ notation)

## What does it provide?
GenSn provides a handful of useful utilities and modules to easily construct and combine probabilistic modeling components. At the very top level, GenSn can be thought of as an extension to `torch.distributions`, providing a way to define *trainable distributions*. Trainable distributions are much like a hybrid of PyTorch modules and PyTorch distribution objects. It let's you easily define a distribution, complete with its trainable parameters.