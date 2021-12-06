# VGG

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

# 코드 구현

[](https://github.com/Soohyeon-Bae/VGG/tree/master)

![그림1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0e260519-181d-4b0a-806e-95b2e55cdaca/그림1.png)

![그림2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d05f1253-660f-44c1-9c64-ef0c7ff811f0/그림2.png)

- 참고
    
    [초보자를위한 Keras의 단계별 VGG16 구현](https://ichi.pro/ko/chobojaleul-wihan-keras-ui-dangyebyeol-vgg16-guhyeon-184940327513708)
    
    [06. 합성곱 신경망 - Convolutional Neural Networks](https://excelsior-cjh.tistory.com/180)
    
    [[DL] LeNet-5, AlexNet, VGG-16, ResNet, Inception Network](https://wooono.tistory.com/233)
    

# 논문 리뷰

# Abstract

- Convolutional Network의 depth가 accuracy에 영향을 줌
- 아주 **작은 conv filter**(3x3)을 사용하여 **depth의 더 깊게** 만들자! → 정확도 개선

# Introduction

성능향상을 위한 방법

- smaller window size and smaller stride(Zeiler & Fergus, 2013; Sermanet et al., 2014)
- multi-scale images (Sermanet et al., 2014; Howard, 2014)
- 이 논문에서는 네트워크의 깊이(**depth**)를 ConvNet 구조의 중요한 측면으로 봄

# ConvNet Configuration

## Architecture

Input

1. Input image size : 224x224 (RGB)
2. Preprocessing : 각 픽셀에서 RGB의 평균 값을  빼줌
    - Why does VGG19 subtract the mean RGB values of inputs?
        - Gradient stability 등을 이유로 계산 상 작은 값을 가지는 편이 알고리즘에 효율적이므로 RGB 평균값을 빼줌으로써 전체적인 RGB 값을 작게 만들어줌
        - Dataset bias 문제를 해결하기 위해서는 평균이 0이고 분산이 1이어야하는데 그렇지 않은 경우이를 보상하기 위한 가중치와 편향이 필요하게 됨(값이 크면 더 심해짐)
        - 모델 training 과정에서 데이터가 정규화/scaled 되어있지 않으면 적절하지 못한 영향을 줄 수 있는데 이를 해결하기 위해서는 값을 작게 만들어주는 것이 좋음

Conv Layers

1. 3x3 필터를 사용하는 이유는 left/right, up/down, center를 표현할 수 있는 최소 크기이기 때문
2. 선형 변환을 위해 1x1 필터도 사용함
3. Stride는 1이고, pooling은 공간 해상도를 유지하기 위해 사용

Pooling Layer

1. Max-pooling layer가 어떤 conv layers 뒤에 5번 나타남
2. 윈도 사이즈는 2x2 픽셀이고 stride는 2

FC Layer

1. 서로 다른 깊이를 가진 stack 다음에 3개의 fully-connected layer가 나타남
2. 처음 두 FC layer는 4096 채널 수를 가짐
3. 마지막 레이어는 1000 채널 수를 가지는데 이는 ILSVRC 분류가 1000개의 클래스를 갖기 때문
4. 마지막 레이어는 활성화 함수로 softmax 사용
- 모든 은닉층의 활성화 함수는 ReLU
- AlexNet에서 사용된 LRN은 연산 시간과 메모리는 많이 들고 성능 향상은 없어서 사용하지 않음

## Configurations

유일한 차이점은 **depth**

- conv3은 3x3필터를, conv1은 1x1필터를 의미
- conv3-N에서 N은 필터의 개수에 해당 (conv3–64는 64개의 3x3 필터를 매개변수로 사용)
- Conv layer를 지날수록 이미지 크기는 작아지고 채널 수는 많아짐
    
    **각 채널은 서로 다른 특징(feature)을 적절히 추출**하도록 학습되므로, 다양한 특징들을 조합하여 적절히 분류를 수행하게 된다.
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd5d7ad7-1afc-4572-8722-db5e16f3b0f9/Untitled.png)

Stride가 1일 때, 3x3 conv 필터링을 3번 반복한 특징맵은 7x7 Receptive field의 효과와 같다.

(CNN에서 **Receptive field**는 각 단계의 입력 이미지에 대해 하나의 필터가 커버할 수 있는 이미지 영역의 일부)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df2bbc4e-69c0-42b3-8bb9-4154797df3cf/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1361fcee-50d1-4a48-8d7e-32c1494be981/Untitled.png)

- 왜 7x7 필터링 한 번 대신 3x3 필터로 convolution 연산을 3번 수행하는 것일까?
    - 비선형성 증가 : 각 conv 연산에는 ReLU 함수를 포함하는데, 1-layer 7x7 필터링은 비선형 함수가 한 번 적용되지만, 3-layer 3x3 필터링은 세 번의 비선형 함수가 적용됨
    - 학습 파라미터 수 감소 : Convolutional Network 구조를 학습할 때, 학습되는 가중치는 필터의 크기에 해당되므로, 7x7필터에 대한 학습 파라미터 수는 49이고 3x3 필터 3개에 대한 학습 파라미터 수는 27임
    
    파라미터 수가 많으면 오버피팅이 일어나기 쉽기 때문에 파라미터 수는 줄이는 것이 좋으나, 항상 층을 깊게 만드는 것이 유리한 것은 아니다. 여러 레이어를 거쳐 만들어진 특징 맵(Feature Map)은 동일한 Receptive Field에 대해 더 추상적인 정보를 담게 된다. 목적에 따라서는 더 선명한 특징 맵이 필요할 수도 있다.
    
- VGG C 모델에서는 1x1 conv layer를 통해 비선형성(non-linearity) 강화
    1. Channel reduction : 특징맵의 크기는 그대로이고 채널 수만 감소
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7584c1d1-26ed-4f15-bdf8-a6ed2e2befec/Untitled.png)
        
    2. 1x1 conv 연산 후에 포함되는 ReLU 함수를 통해 비선형성 추가

# Classification Framework

## Training

하이퍼 파라미터 설정

- Multinomial logistic regression objective
- Mnin-batch size : 256
- Momentum : 0.9
- L2 regularization : 5x10^(-4)
- Dropout : 0.5
- Learning rate : 10^(-2), 검증 데이터셋의 정확도가 높아지는 것을 멈추면 10배씩 감소

깊은 네트워크와 많은 파라미터에도 불구하고 더 적은 에폭(74 epochs)이 필요한 이유는 

(a) 깊고 작은 conv 필터에 의한 implicit regularization 

(7x7 필터를 사용할 때보다 3x3 필터를 사용하면 파라미터가 적어짐: implicit regularization,

반대로 dropout을 사용하는 방식은 explicit regularization이라고 할 수 있다.)

(b) 특정 레이어에서의 pre-initialization

(VGG A모델을 학습하고 B,C,D,E 모델을 구성할 때, 이미 이전 모델에서 학습된 layer를 가져다 씀)

깊은 네트워크를 만들 때, VGG A모델에서 처음 4개 conv layers와 마지막 3개 FC layer를 가져다가 초기화에 사용하여 최적의 초기값을 설정해주어 학습이 용이하게 함

224x224 input images의 증강을 위해 무작위 cropping, flipping, color shifting을 진행함

### Training image size

$S$ : the smallest side of an isotropically-rescaled training image

(Training image의 높이와 너비 중 더 작은 쪽을 256으로 줄이되, aspect ratio를 유지하면서 나머지 쪽도 rescale하는 방식)

이후, rescale된 이미지를 224x224로 crop

$**S$ 값을 설정하는 두가지 방식**

1. Single-scale training : $S$값을 고정
    - 먼저 $S$ = 256으로 모델을 학습시킨 다음, 학습 속도를 줄이기 위해 앞서 학습된 가중치를 사용하여 $S$ = 384로 추가학습(lr 줄여서)
2. Multi-scale training : S 값을 256에서 512 사이의 값으로 랜덤하게 결정
    - 이미지의 개체는 크기가 다를 수 있으므로 훈련 중에 이를 고려하도록 하면 학습 효과가 좋아질 수 있음
    - Scale jittering에 의한 훈련 데이터 증강
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7127ae8f-ac61-4b9a-9da4-851b3102255e/Untitled.png)
        
    - 속도를 높이기 위해 앞서 single-scale training으로 사전학습된 모델을 fine-tuning하여 진행

## Testing

$Q$  :  isotropically rescaled to a pre-defined smallest image side

- $Q = S$ 일 필요는 없으며, $S$ 값 마다  다른 $Q$를 적용하면 성능이 향상됨
- 테스트할 때는 마지막 3개의 FC layers를 conv layers로 변환(fully-convolutional layers)
- 첫번째 FC layer는 7x7 conv layer로, 다음 2개의 FC layers는 1x1 conv layers로 바꿔 사용함
    
    (첫 번째 FC layer를 (7x7x512) 4096 필터의 conv layer로 변경하면 가중치의 수가 유지된다.)
    

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/daa6f7be-201b-4aa4-be46-8e09c51c40a9/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d96c865f-a5d4-4076-9dda-0ecef0f69a12/Untitled.png)

FC layer는 입력 노드가 하이퍼 파라미터로 정해져 있기 때문에 입력 사이즈가 고정되어야 하지만, conv layer는 입력 사이즈의 제약이 없음

이로 인해 , 테스트 시에는 uncropped된 whole images도 사용할 수 있음

그러나 입력 이미지가 커지면 출력된 특징맵(class score map)이 1x1보다 크게 될 수도 있음

7x7의 class score map은 1000 채널을 가지고, 각 7x7 특징맵을 average pooling하는 작업 수행

![Global Average Pooling](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89407fe6-da5c-4ac8-b98d-3cc86de2978a/Untitled.png)

Global Average Pooling

softmax 이후에 flipped image와 original image의 평균을 내어 최종 score 출력

# Classification Experiments

- **Top-1 error** : Multi-class classification error로 **잘못 분류된 image의 비율**
- **Top-5 error** : **모델이 예측한 최상위 5개 범주 안에 정답이 없는 경우**

## Single-Scale Evaluation

테스트 이미지 크기

1. 고정된 $S$ 값에 대해서, $Q = S$   
2. Scale jittering을 통해 결정된 $S ∈ [S_{min}, S_{max}]$에 대해,  $Q = 0.5(S_{min} + S_{max})$ 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12793b08-5e90-4b65-b059-e778cfbb2ed9/Untitled.png)

- Classifiaciton error는 네트워크의 깊이가 늘어날수록 줄어듦
- 3개의 1x1 필터를 포함한 C모델은 3x3 필터를 사용한 D모델보다 성능이 나쁨

C모델이 B모델보다 성능이 나은 것을 통해 비선형성을 추가하는 것이 좋다는 것을 알 수 있지만, D모델이 C모델보다 성능이 좋은 것은  3x3 필터가 위치 정보 특징을 더 잘 추출하기 때문임

- ILSVRC 데이터셋에서는 레이어를 19층 이상으로 쌓았을 때 추가적인 성능 향상은 없었지만, 더  큰 데이터셋에서는 성능이 향상될 수도 있음
- 5x5 필터를 사용한 네트워크와 B모델(3x3 필터 사용)의 비교를 통해 **작은 필터를 사용한 깊은 네트워크**가 큰 필터를 사용한 얕은 네트워크보다 성능이 더 뛰어남을 확인함
- 훈련 과정에서의 **scale jittering은 데이터 증강 효과**를 가져와서 고정된 이미지 크기를 사용하였을 때보다 성능이 좋음

## Multi-Scale Evaluation

- 테스트 이미지의 여러 가지 크기 조정(다양한 $Q$ 값)에 대해 클래스 평균 계산
- 훈련과 테스트 이미지 스케일이 다를 때 나타나는 성능 저하를 고려하고자, 고정된 $S$ 값을 사용한 경우,  $Q = [S − 32, S, S + 32]$로 설정
- 훈련 과정에서 scale jittering을 적용한 경우($S ∈ [S_{min}; S_{max}]$),  $Q = [{S_{min}, 0.5(S_{min} + S_{max}), S_{max}}]$로 설정

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/72641941-7e51-4a63-b7e6-c43514adcb1e/Untitled.png)

- Scale jittering을 적용한 경우에 더 성능이 좋음
- 가장 깊은 네트워크를 가진 D모델과 E모델의 성능이 가장 좋음

## Multi-crop Evaluation

- Dense evaluation
    - FC layer를 1x1 conv layer로 바꿔 사용하는 방식
    - 큰 영상에 대해 ConvNet을 적용하고 일정한 픽셀 간격(grid)으로 결과를 끌어냄 (sliding window와 비슷)
    - 연상량을 효율적으로 줄일 수 있음
    - Grid 크기 문제로 인해 학습 결과가 떨어질 수 있음
- Multi-crop evaluation
    - 원 영상으로부터 영상을 잘라낸 후 각각 ConvNet에 적용
- Multi-crop 방식이 dense evaluation보다 살짝 성능이 높지만, 둘은 상호보완적 관계

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8bc7e40d-7962-4e13-964e-7e8b5d2eb207/Untitled.png)

## ConvNet Fusion

- 7개의 모델을 ensemble한 결과보다 D모델과 E모델만 ensemble한 결과가 더 좋음

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f1dcafa7-6c68-421c-b3a6-fda872c029d7/Untitled.png)

## Comparison with the State Of The Art

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f352ee0e-be92-47ba-bd20-a104dc56af66/Untitled.png)

# Conclusion

> Our results yet again confirm the importance of depth in visual representations.
> 

- 참고
    
    [https://89douner.tistory.com/61](https://89douner.tistory.com/61)
    
    [Why does VGG19 subtract the mean RGB values of inputs?](https://stackoverflow.com/questions/48201735/why-does-vgg19-subtract-the-mean-rgb-values-of-inputs)
    
    [VGG16 논문 리뷰 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a)
    
    [CNN (Convolutional Neural Network) 요약 정리](https://ndb796.tistory.com/477)
    
    [1.Deep Neural Network가 이미지영역에서는 성과가 좋지 않다구요? (Convolution Neural Network)](https://89douner.tistory.com/55)
    
    [FCN 논문 리뷰 - Fully Convolutional Networks for Semantic Segmentation](https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204)
    
    [[Part Ⅴ. Best CNN Architecture] 6. VGGNet [2] - 라온피플 머신러닝 아카데미 -](https://m.blog.naver.com/laonple/220749876381)
    
    [[Part Ⅴ. Best CNN Architecture] 7. OverFeat - 라온피플 아카데미 -](https://m.blog.naver.com/laonple/220752877630)
