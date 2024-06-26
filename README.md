## Deep Seam Prediction for Image Stitching Based on Selection Consistency Loss
<br> ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) 
<img src="https://img.shields.io/badge/Anaconda-44A833?style=flat-square&logo=Anaconda&logoColor=white"/>
<br>
<br>
 ![image](https://github.com/YOOSUSANG/Deep-Seam-Prediction/assets/41133135/d049ca70-0faa-4077-b6d3-694dd66c98ba)
<br>
### 프로젝트 개요
이미지 스티칭 기술은 다양한 분야에서 활용되고 있지만, 저희 팀은 현재 AVM(주변 시야 모니터링 시스템) 프로젝트를 주로 다루고 있습니다. AVM은 차량과 선박 등에 설치된 여러 카메라로부터 얻은 영상을 통합하여, 전방, 후방, 측면 등 다양한 각도에서의 시야를 제공합니다. 이 시스템은 주변 장애물, 보행자, 차선 등을 정확히 감지하여 운전자에게 안전성을 높이고 운전 편의성을 증진시키는 목적으로 개발되었습니다.

여러 카메라로부터 얻은 영상에 대한 시차가 작을수록 이음새 부분에서 발생하는 고스트 현상이 적습니다. 그러나 선박과 대형 차량과 같은 경우 시차가 크기 때문에 고스트 현상 발생 확률이 높습니다. 현재 OpenCV에서는 dynamic programming과 GraphCut 알고리즘이 있지만, 여러 카메라 영상을 실시간으로 통합하여 보여주는 데 어려움이 있습니다.

이러한 문제를 해결하기 위해 딥러닝 기반의 고성능 품질 네트워크인 DSeam을 사용하여 큰 시차에서도 이음새를 정밀하게 예측하고 통합할 수 있습니다. 이 방법을 통해 운전자에게 실시간으로 정확한 시야를 제공하여 안전성과 편의성을 보장할 수 있습니다.

이 프로젝트는 기존 회사 프로그램과 성능을 비교하기 위한 연구 목적으로 (주)큐램에 소속되어 있는 최나람 연구원님의 지시를 받아 독립적으로 개발을 진행하였습니다.

### 프로젝트 필요성
차량 AVM의 경우 시차가 크지 않아 상당히 많이 사용되고 있습니다. 그러나 선박과 같이 시차가 큰 경우 AVM을 적용하는 것은 제약 사항이 많고 어렵습니다. 이 문제를 해결하기 위해 DSEAM 네트워크가 필요합니다. DSEAM 네트워크는 실시간으로 뛰어난 성능을 발휘하며 이음새의 품질을 향상시킬 수 있습니다.

만약 DSEAM 네트워크가 선박 운항에서 실제로 이런 성과를 낼 수 있다면, 하이리턴 하이리스크였던 선박 운항이 하이리턴 로우리스크로 전환될 수 있음을 확신할 수 있습니다.

### 프로젝트 결과
![그림5 (1)](https://github.com/YOOSUSANG/Deep-Seam-Prediction/assets/41133135/d962c24d-17fb-4653-b0e5-7611a6c4e635)



**nonoverlap**

![그림1 (1)](https://github.com/YOOSUSANG/Deep-Seam-Prediction/assets/41133135/68b49960-7054-4200-ba28-ca647147ccf4)


**overlap**

![그림2 (2)](https://github.com/YOOSUSANG/Deep-Seam-Prediction/assets/41133135/2c91809d-0dd2-454b-87e9-8e9b22a2bc56)



### 문제점
학습 중에 loss가 더이상 감소하지 않는 문제가 발생했습니다. 혼자서 시도하다보니 몇 가지 문제점을 발견했는데, 주로 입력 이미지 처리나 loss 계산 식이 모호할 가능성이 큽니다. 또는 epoch 수가 충분하지 않아서 문제가 발생할 수도 있습니다.

### checkpoint
[https://drive.google.com/drive/folders/1F7A-aJzc6g6NtBtRoyRoXqoa5SXAXA7M](https://drive.google.com/drive/folders/1F7A-aJzc6g6NtBtRoyRoXqoa5SXAXA7M)

### 참고
[U-Net: Semantic segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet)

[Deep Seam Prediction for Image Stitching Based on Selection Consistency Loss](https://arxiv.org/abs/2302.05027)


[Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images (UDIS)](https://github.com/nie-lang/UnsupervisedDeepImageStitching)
