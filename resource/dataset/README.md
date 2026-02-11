---
license: mit
size_categories:
- 100K<n<1M
---

# commaVQ
commaVQ is a dataset of 100,000 heavily compressed driving videos for Machine Learning research. A heavily compressed driving video like this is useful to experiment with GPT-like video prediction models. This repo includes an encoder/decoder and an example of a video prediction model.
Examples and trained models can be found here: https://github.com/commaai/commavq

# Overview
A VQ-VAE [1,2] was used to heavily compress each frame into 128 "tokens" of 10 bits each. Each entry of the dataset is a "segment" of compressed driving video, i.e. 1min of frames at 20 FPS. Each file is of shape 1200x8x16 and saved as int16.

Note that the compressor is extremely lossy on purpose. It makes the dataset smaller and easy to play with (train GPT with large context size, fast autoregressive generation, etc.). We might extend the dataset to a less lossy version when we see fit.

<video title="source" controls>
  <source src="https://github.com/commaai/commavq/assets/29985433/91894bf7-592b-4204-b3f2-3e805984045c" type="video/mp4">
</video>

<video title="compressed" controls>
  <source src="https://github.com/commaai/commavq/assets/29985433/3a799ac8-781e-461c-bf14-c15cea42b985" type="video/mp4">
</video>

<video title="imagined" controls>
  <source src="https://github.com/commaai/commavq/assets/29985433/f6f7699b-b6cb-4f9c-80c9-8e00d75fbfae" type="video/mp4">
</video>

# References
[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.