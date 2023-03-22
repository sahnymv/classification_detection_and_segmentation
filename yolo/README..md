# Paper Summary
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
## Related Works
- Current detection systems repurpose classifiers to perform detection.
- Deformable Parts Models (DPM) [10]
    - Systems like deformable parts models (DPM) use a sliding window approach where the classifier is run at evenly spaced locations over the entire image.
- R-CNN [13]
    - More recent approaches like R-CNN use region proposal methods to first generate potential bounding boxes in an image and then run a classifier on these proposed boxes. After classification, post-processing is used to refine the bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene.
- ***These complex pipelines are slow and hard to optimize because each individual component must be trained separately.***
- A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. ***Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.***
- Fast R-CNN [14]
    - ***Fast R-CNN, a top detection method, mistakes background patches in an image for objects because it can’t see the larger context.***
    - YOLO makes less than half the number of background errors compared to Fast R-CNN.
## Methodology
- YOLO still lags behind state-of-the-art detection systems in accuracy. ***While it can quickly identify objects in images it struggles to precisely localize some objects, especially small ones. Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an image simultaneously.*** The YOLO design enables end-to-end training and real-time speeds while maintaining high average precision.
- ***Our system divides the input image into an*** $S \times S$ ***grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts*** $B$ ***bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.***
- Formally we define confidence as
$$P(Object) \cdot IOU^{gt}_{pred}$$
- ***If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.***
- ***Each bounding box consists of 5 predictions: x, y, w, h and confidence.***
    - The $(x, y)$ coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. ***We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.***
- Finally the confidence prediction represents the IOU between the predicted box and any ground truth box. Each grid cell also predicts $C$ conditional class probabilities, $P(Class \mid Object)$. ***We only predict one set of class probabilities per grid cell.*** It divides the image into an $S \times S$ grid and ***for each grid cell predicts*** $B$ ***bounding boxes, confidence for those boxes, and*** $C$ ***class probabilities. These predictions are encoded as an*** $S \times S \times (B \times 5 + C)$ ***tensor.***
- Non maximum suppression
    - The grid design enforces spatial diversity in the bound- ing box predictions. Often it is clear which grid cell an object falls in to and the network only predicts one box for each object. However, some large objects or objects near the border of multiple cells can be well localized by multiple cells. Non-maximal suppression can be used to fix these multiple detections. ***While not critical to performance as it is for R-CNN or DPM, non-maximal suppression adds 2-3% in mAP.***
## Architecture
- Figure 2
    - <img src="https://user-images.githubusercontent.com/105417680/226794538-894211b8-9841-459b-bb59-a4d38e85cbe6.png" width="600">
- Figure 3. YOLO architecture
    - <img src="https://user-images.githubusercontent.com/105417680/226794555-718d0bf8-48e6-489f-a78d-b6b1bae4edd8.png" width="800">
    - The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates. Our network architecture is inspired by the GoogLeNet model for image classification [34].
    - Our network has 24 convolutional layers followed by 2 fully connected layers. Instead of the inception modules used by GoogLeNet, we simply use $1 \times 1$ reduction layers followed by $3 \times 3$ convolutional layers, similar to [22].
- We use a linear activation function for the final layer and all other layers use rectified linear activation with slope coefficient of $0.1$.
- Our final layer predicts both class probabilities and bounding box coordinates.
- Fast YOLO
    - We also train a fast version of YOLO designed to push the boundaries of fast object detection. Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) and fewer filters in those layers. Other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO.
- A dropout layer with $rate = 0.5$ after the first connected layer prevents co-adaptation between layers [18].
### Loss
$$
\lambda_{coord} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \mathbb{1}^{obj}_{ij} \bigg[(x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} + (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2} + (C_{i} - \hat{C}_{i})^{2}\bigg]
\\+ \lambda_{noobj} \sum^{S^{2}}_{i = 0} \sum^{B}_{j = 0} \big(C_{i} - \hat{C}_{i}\big)^{2}
\\+ \sum^{S^{2}}_{i = 0} \mathbb{1}^{obj}_{i} \sum_{c \in classes} \big(p_{i}(c) - \hat{p}_{i}(c)\big)^{2}
$$
- We use sum-squared error because it is easy to optimize, however ***it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the "confidence" scores of those cells towards zero, often overpowering the gradient from cells that do contain objects.*** This can lead to model instability, causing training to diverge early on. To remedy this, ***we increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects. We use two parameters,*** $\lambda_{coord}$ ***and*** $\lambda_{noobj}$ ***to accomplish this. We set*** $\lambda_{coord} = 5$ ***and*** $\lambda_{noobj} = 0.5$$$***.***
- Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly. At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be "responsible" for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall. During training we optimize the following, multi-part
- $\mathbb{1}^{obj}_{i}$ ***denotes if object appears in cell*** $i$ ***and*** $\mathbb{1}^{obj}_{ij}$ ***denotes that the*** $j$th ***bounding box predictor in cell*** $i$ ***is "responsible" for that prediction.***
- ***Note that the loss function only penalizes classification error if an object is present in that grid cell. It also only penalizes bounding box coordinate error if that predictor is "responsible" for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).***
## Training
### Pre-training
- Dataset
    - ImageNet 1000-class competition dataset [30]
- ***We pre-train the convolutional layers at half the resolution (***$224 \times 224$ ***input image) and then double the resolution for detection.***
- ***For pretraining we use the first 20 convolutional layers from Figure 3 followed by a average-pooling layer and a fully connected layer.***
### Fine-tuning
- Dataset
    - We train the network for about 135 epochs on the training and validation data sets from PASCAL VOC 2007 and 2012. When testing on 2012 we also include the VOC 2007 test data for training.
- Throughout training we use a batch size of 64, ***a momentum of 0.9 and a decay of 0.0005***. Our learning rate schedule is as follows: ***For the first epochs we slowly raise the learning rate from*** $10^{−3}$ ***to*** $10^{−2}$***. If we start at a high learning rate our model often diverges due to unstable gradients. We continue training with*** $10^{−2}$ ***for 75 epochs, then*** $10^{−3}$ ***for 30 epochs, and finally*** $10^{−4}$ ***for 30 epochs.***
- Data augmentation
    - We introduce random scaling and translations of up to 20% of the original image size. We also randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.
## Evaluation
- For evaluating YOLO on PASCAL VOC, we use $S = 7$, $B = 2$. PASCAL VOC has 20 labelled classes so $C = 20$. Our final prediction is a $7 \times 7 \times 30$ tensor. We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset [9].
## References
- [9]
- [10] [Object Detection with Discriminatively Trained Part Based Models](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)
- [13] [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
- [14] [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- [18] [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf)
- [22] [Network in network](https://arxiv.org/pdf/1312.4400.pdf)
- [26] [Darknet: Open Source Neural Networks in C](https://pjreddie.com/darknet/)
- [34] [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)