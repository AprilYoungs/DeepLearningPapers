
# 1. Survey and Tutorials
## 1. Summarization
1. Recent Advances in Deep Learning An Overview(2018.07)
1. A Survey on Deep Learning Algorithms, Techniques,and Applications(2018)
1. deep learning(2015)
2. Three Classes of Deep Learning  Architectures and Their Applications: A Tutorial Survey(2014)

## 2. Tutorials
1. [A guide to convolution arithmetic for deep learning(2016.03)](https://arxiv.org/abs/1603.07285)
2. [Deep learning:Technical introduction(2017.09)](https://arxiv.org/abs/1709.01412)
3. [Deep Learning Tutorial(2015)](http://deeplearning.net/tutorial/deeplearning.pdf)
4. [Notes on Convolutional Neural Networks](https://github.com/TimeFunny/DeepLearningPapers/blob/master/Tutorials/Notes%20on%20Convolutional%20Neural%20Networks.pdf)
5. Notes on Backpropagation

# 2. Architecture
## 1. CNN
1. AlexNet: ImageNet Classification with Deep Convolutional
Neural Networks
1. One weird trick for parallelizing convolutional neural networks
2. VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition
3. Inceptions-v1: Going deeper with convolutions
4. Inceptions-v2: Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift
5. Inceptions-v3: Rethinking the Inception Architecture for Computer Vision
6. Inceptions-v4: Inception-ResNet and the Impact of Residual Connections on Learning
7. ResNet: Deep Residual Learning for Image Recognition
8. Identity Mappings in Deep Residual Networks
8. ResNeXt: ggregated Residual Transformations for Deep Neural Networks
9. Xception: Deep Learning with Depthwise Separable Convolutions
10. DenseNet: Densely Connected Convolutional Networks
11. DPN: Dual Path Networks
12. SENet: Squeeze-and-Excitation Networks
---
1. An intriguing failing of convolutional neural networks and the CoordConv solution(2018.07)
14. SkipNet: Learning Dynamic Routing in Convolutional Networks(2017.11)
14. Gradually Updated Neural Networks for Large-Scale Image Recognition(2017.11)
14. NASNet: Learning Transferable Architectures for Scalable Image Recognition(Neural Architecture Search 2017.07)
14. Residual Attention Network for Image Classification(2017.04)
14. Deformable Convolutional Networks(2017.03)
15. Spatial Transformer Networks(2015.06)
---
### 移动端

1. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applicatio
14. MobileNet-v2: Inverted Residuals and Linear Bottlenecks(2018.01)
15. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile
16. ShuffleNet-V2: Practical Guidelines for Efficient CNN Architecture Design(2018.07)
17. A Quantization-Friendly Separable Convolution for MobileNets(2018.03)

## 2. RNN(欢迎提供相关资料)
1. LSTM: [LSTM wiki](https://en.wikipedia.org/wiki/Long_short-term_memory)
2. GRU: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
3. BILSM: Bidirectional Recurrent Neural Networks(1997)
4. Recent Advances in Recurrent Neural Networks(2018.01)


# 3. object detection
1. Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks(2018)  
2. Deep Learning for Generic Object Detection: A Survey(2018)
[github](https://github.com/hoya012/deep_learning_object_detection)
3. [目标检测综述一](https://mp.weixin.qq.com/s/Xw7wwKLijzfFYHPFdgRP2A)
4. [目标检测综述二](https://mp.weixin.qq.com/s/ymU571dD_3l49blEFo3VsA)

## 1. 2D detection
1. CornerNe Detecting Objects as Paired Keypoints.pdf(2018.08)
2. **IoU-Net**:Acquisition of Localization Confidence for Accurate Object Detection(2018.07)
3. CFENet: An Accurate and Efficient Single-Shot Object Detector for Autonomous Driving(2018.06)
4. Object detection at 200 Frames Per Second(2018.05)
6. **RefineDet**:Single-Shot Refinement Neural Network for Object Detection(cvpr2018)
7. **Pelee**: A Real-Time Object Detection System on Mobile Devices(2018.04)
4. DetNet: A Backbone network for Object Detection(2018.04))
5. Cascade R-CNN Delving into High Quality Object Detection(2017.12)
6. An Analysis of Scale Invariance in Object Detection – SNIP(2017.11) [code](https://github.com/bharatsingh430/snip)
7. DSOD: Learning Deeply Supervised Object Detectors from Scratch(2017.08)
8. Soft-NMS Improving Object Detection With One Line of Code(2017.04)
8. DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling(2017.03)
9. Speed/accuracy trade-offs for modern convolutional object detectors(2016.11)
10. R-FCN Object Detection via Region-based Fully Convolutional Networks
---
### 经典
#### 1.two-stage
1. Fast R-CNN:
2. Faster R-CNN:Towards Real-Time Object Detection with Region Proposal Networks
3. An Implementation of Faster RCNN with Study for Region Sampling
3. FPN:Feature pyramid networks for object detection
4. RetinaNet:Focal Loss for Dense Object Detection [Detectron Code caffe2](https://github.com/facebookresearch/Detectron)  

#### 2.one-stage
1. YOLO v1:You Only Look Once:Unified, Real-Time Object Detection
3. YOLO v2:YOLO9000 Better, Faster, Stronger
4. YOLO v3:An Incremental Improvement
---
1. SSD:Single Shot MultiBox Dete
2. DSSD: Deconvolutional Single Shot Detector
---
## 2. 3D detection(欢迎提供相关资料)
1. YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud(2018.08)
2. 2D-Driven 3D Object Detection in RGB-D Images(ICCV2017)
2. 3D Object Proposals for Accurate Object Class Detection(KITTI Dataset,2015)

[//]: #(应该可以归类为跟踪问题)
<!--
## 3. Video
2. T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos(2016.04)
3. Seq-NMS for Video Object Detection(2016.02)
-->

# 4. segmentation
1. A survey on deep learning techniques for image and video semantic segmentation(2016.04)

## 1.Image
### 1. Dataset
1. COCO-Stuff: Thing and Stuff Classes in Context(uncountable)
2. Microsoft COCO: Common Objects in Context
3. The PASCAL Visual Object Classes (VOC) Challenge

### 2. Semantic segmentation
1. ICNet for Real-Time Semantic Segmentation on High-Resolution Images(ECCV2018)
2. **DANet**: Dual Attention Network for Scene Segmentation(2018.09)
3. **EncNet**:Context Encoding for Semantic Segmentation(2018.03)
3. Loss Max-Pooling for Semantic Image Segmentation(2017.04)
4. Understanding Convolution for Semantic Segmentation(2017.02)
5. Multi-Scale Context Aggregation by Dilated Convolutions(2015.11)

#### PSPNet
1. Pyramid Scene Parsing Network(2016.12)

#### U-Net
1. U-Net Convolutional Networks for Biomedical(2015)
#### SegNet
1. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation(2015)
#### FCN
1. Fully Convolutional Networks for Semantic Segmentation(2016)

---
1. Semantic Soft Segmentation(2018.08)
2. Deep Image Matting(2017.03)
---
#### DeepLab
1. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs(2014)
2. [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs(2016.06)](http://liangchiehchen.com/projects/DeepLab.html)
3. DeepLab v3: Rethinking Atrous Convolution for Semantic Image Segmentation(2017.06)
4. DeepLab V3+: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation(2018.02)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)

### 3. Instance segmentation
1. Faster Training of Mask R-CNN by Focusing on Instance Boundaries(2018.09)
2. Instance Segmentation by Deep Coloring(2018.07)
2. Learning to Segment Every Thing(2017.11)
3. Mask R-CNN
4. Fully Convolutional Instance-aware Semantic Segmentation(2016.11)
5. Instance-aware Semantic Segmentation via Multi-task Network Cascades(2015.12)
6. depth-aware object instance segmentation (2015.12)
6. Monocular Object Instance Segmentation and Depth Ordering with CNNs(2015.05)

### 4. Panoptic Segmentation
1. Panoptic Segmentation(Survey)  (2018.01)
 
### 5. 3D
1. Segmenting Unknown 3D Objects from Real Depth Images using Mask R-CNN Trained on Synthetic Point Clouds(2018.09)

## 2. Video
1. [Mobile Real-time Video Segmentation](https://ai.googleblog.com/2018/03/mobile-real-time-video-segmentation.html)
2. MaskRNN: Instance Level Video Object Segmentation(NIPS 2017)


# 5. Object Tracking

## 1.  Survey
1. [Object Tracking: A Survey(2006)](http://vision.eecs.ucf.edu/projects/trackingsurvey/)
2. [New Trends on Moving Object Detection in Video Images Captured 
by a moving Camera: A Survey ](https://hal.archives-ouvertes.fr/hal-01724322/document)
3. [Tracking Noisy Targets: A Review of Recent Object
Tracking Approaches](https://arxiv.org/pdf/1802.03098.pdf)
4. A Survey on Object Detection and Tracking Algorithms(2013)

## 2. VOT
1. The Visual Object Tracking VOT2017 challenge results(2017)
---
1. Tracking Emerges by Colorizing Videos(2018.06)
2. A Twofold Siamese Network for Real-Time Object Tracking(2018.02)
3. Triplet Loss in Siamese Network for Object Tracking(ECCV2018)
3. End-to-end representation learning for Correlation Filter based tracking(2017)
4. Siamese Learning Visual Tracking: A Survey(2017.07)
4. Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects(2017.05)
5. Fully-Convolutional Siamese Networks for Object Tracking(2016.06)
6. Robust Visual Tracking with Deep Convolutional Neural Network based Object Proposals on PETS(CVPR 2016)
7. Visual Object Tracking using Adaptive Correlation Filters


10. Certain Approach of Object Tracking using Optical Flow Techniques(2012)
## 3. MOT
1. Real-Time Multiple Object Tracking A Study on the Importance of Speed(2017.09)
2. Online Multi-Object Tracking Using CNN-based Single Object Tracker with Spatial-Temporal Attention Mechanism(2017)
3. Multiple-object tracking while driving: the multiple-vehicle tracking task(2014)
 
# Face Tasks(欢迎提供相关资料)
## 1.Detection
1. PyramidBox A Context-assisted Single Shot Face Detector(2018.03)
2. A Convolutional Neural Network Cascade for Face Detection(2015)
4. Rapid Object Detection using a Boosted Cascade of Simple Features(2001)

## 2.Keyponits
1. Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet(2017.10)

## 3.Recognition
1. Deep Face Recognition: A Survey(2018.04)
1. multi-task learning for face identification and attribute estimation(ICASSP 2017)

3. Facenet: A unified embedding for face recognition and clustering(2015.03) 

## 4.Retrieval
1. Face Video Retrieval via Deep Learning of Binary Hash Representations(2016) 



# Pedestrian(欢迎提供相关资料)
1. In Defense of the Triplet Loss for Person Re-Identification(2017.03)


# Person Pose(欢迎提供相关资料)
1. Simple Baselines for Human Pose Estimationand Tracking(2018)
2. MultiPoseNet Fast Multi-Person Pose Estimation using Pose Residual Network(2018)
3. PoseTrack: Joint Multi-Person Pose Estimation and Tracking(2017)
6. DeepPose: Human Pose Estimation via Deep Neural Networks(2014)
 
## 1. keyponit
1. [Realtime Multi-Person 2D Pose Estimation using Part Affinity Field(2016.11)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)



# OCR
## 1. Detection
1. EAST An Efficient and Accurate Scene Text Detector(2017)


## 2. Recognition
1. Connectionist Temporal Classification Labelling Unsegmented Sequence Data with Recurrent Neural Network(2006,可变字符序列识别)
2. mutilple object recognition with visual attention(2014,可变字符序列识别)

### Rotation
1. [Arbitrary-Oriented Scene Text Detection via Rotation Proposals(2017.03)](https://github.com/mjq11302010044/RRPN)
2. [R2CNN Rotational Region CNN for Orientation Robust Scene Text Detection(2017.06)](https://github.com/yangxue0827/R2CNN_FPN_Tensorflow)

## 3.License Plate
1. LPRNet: License Plate Recognition via Deep Neural Networks(2018.06, 100ms)
2. License Plate Detection and Recognition Using CNN(2017.03)
3. Benchmark for License Plate Character Segmentation(2016.07)
4. [open source code](https://github.com/openalpr/openalpr)


# GAN
1. Generative Adversarial Nets(2014.06)
2. Improved Techniques for Training GANs(2016.06)
3. Generative Adversarial Networks Tutorial(2017.01)

---
1. Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks(2018.09)
2. Enhanced Super-Resolution Generative Adversarial Networks(2018.09)

---
1. CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[coe](https://github.com/junyanz/CycleGAN)
2. pix2pix: Image-to-image translation using conditional adversarial nets[code](https://github.com/phillipi/pix2pix)
3. vid2vid:[code](https://github.com/NVIDIA/vid2vid)



# Training
1. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour(2017)  

## 调参
1. ImageNet Training in Minutes(2017.09)

## 1. Activation/loss function
1. [activation](https://en.wikipedia.org/wiki/Activation_function)
2. Adaptive Sampled Softmax with Kernel Based Sampling(2018)


## 2. Learning rate decay
1. SGDR: Stochastic Gradient Descent with Warm Restarts(cosine schedule)
2. [Blog](https://zhuanlan.zhihu.com/p/32923584)
3. [GluonCV](https://zhuanlan.zhihu.com/p/38509951)


## 3. Regularization
1. ShakeDrop regularization(2018)
2. Shakeout: A New Regularized Deep Neural Network Training Scheme(2016)
3. Dropout: A Simple Way to Prevent Neural Networks from overfitting
 
## 4. Normalization
1. Instance Normalization(2016)

# Image Annotation Tools(欢迎提供相关资料)
1. [Visual Object Tagging Tool](https://github.com/Microsoft/VoTT)
2. [Computer Vision Annotation Tool](https://github.com/opencv/cvat/)
3. [Mask Editor an Image Annotation Tool for Image Segmentation Tasks(2018.09)](https://github.com/Chuanhai/Mask-Editor)
4. A Diverse Driving Video Database with Scalable Annotation Tooling
 
# Open Source CV Repository 
1. [GluonCV(Mxnet)](https://gluon-cv.mxnet.io/model_zoo/classification.html)
2. [Models(tf)](https://github.com/tensorflow/models/tree/master/research/object_detection)
3. [mmdetection(Pytorch)](https://github.com/open-mmlab/mmdetection)


# Image Stylization
1. A Closed-form Solution to Photorealistic Image Stylization(2018)
2. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization(2017)
3. Real-Time Neural Style Transfer for Videos(2017)
3. A Neural Algorithm of Artistic Style(2015)

# Caption
1. Image Captioning with Deep Bidirectional LSTMs(2016.04)
2. Show Attend and Tell Neural Image Caption Generation with Visual Attention(2015.02)
3. Learning CNN-LSTM Architectures for Image Caption Generation
4. Show and Tell: A Neural Image Caption Generator(2014.11)

<!--
# Depth
1. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network(NIPS 2014)
-->














