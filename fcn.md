# Paper Summary
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)

# Convolutionalization
- <img src="https://miro.medium.com/max/1400/1*2IuHjzPjjGXtDU-eAUzxeA.webp" width="600">
- 7x7x512의 Feature map을 4,096차원의 FC layer로 변환하기 위한 가중치의 수: (7 * 7 * 512) * 4096
- 7x7x512의 Feature map을 1x1x4096의 Feature map으로 변환하기 위한 Convolution layer의 가중치의 수: (7x7x512) * 4096
- 즉 동일한 가중치의 수를 갖습니다.

# Deconvolution
- <img src="https://miro.medium.com/max/1400/1*gtBk1yTapyFzvh00DUkGBA.webp" width="600">
