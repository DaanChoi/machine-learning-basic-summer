'''

MLP : https://rubber-tree.tistory.com/139
Cifar10 데이터 로드 : https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
용어 설명 및 Cifar 데이터 설명 : https://gruuuuu.github.io/machine-learning/cifar10-cnn/
※ batch 에러
- https://stackoverflow.com/questions/62157890/how-to-solve-error-no-match-between-expected-input-batch-size-and-target-batch
- https://biology-statistics-programming.tistory.com/239

'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available() # GPU 사용 가능 여부 반환
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 gpu / 아니라면 cpu 사용
print("다음 기기로 학습합니다: ", device)

# reproducibility 재현성
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
learning_rate = 0.001
training_epochs = 15 # 학습 횟수
batch_size = 100 # 연산 한 번에 들어 가는 데이터 크기
drop_prob = 0.3 # dropout 확률

## Dataset 설정 ##
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_train = torchvision.datasets.CIFAR10(root='CIFAR-data/',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)
cifar_test = torchvision.datasets.CIFAR10(root='CIFAR-data/',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

data_loader = DataLoader(dataset=cifar_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## 모델 설계 ##
# MLP이므로 여러 Layer 설정
# cifar10은 32 * 32(1024) 픽셀 크기의 데이터
# tensor의 가로 * 세로 * 채널 => 32 * 32 * 3
# 512는 보통 512로 한다 함
# 최종 결과는 0~9의 정수값 (10개의 클래스)
linear1 = nn.Linear(3072, 512, bias=True) # nn.Linear(input dem, output dem)
linear2 = nn.Linear(512, 512, bias=True)
linear3 = nn.Linear(512, 512, bias=True)
linear4 = nn.Linear(512, 10, bias=True)

# 활성화 함수 ReLU
relu = nn.ReLU()
# dropout 설정
dropout = nn.Dropout(p=drop_prob)

# 가중치(Weight) 초기화
nn.init.xavier_uniform(linear1.weight)
nn.init.xavier_uniform(linear2.weight)
nn.init.xavier_uniform(linear3.weight)
nn.init.xavier_uniform(linear4.weight)

# model 생성
model = nn.Sequential(
    linear1, relu, dropout,
    linear2, relu, dropout,
    linear3, relu, dropout,
    linear4
)

## Cost 함수 & Optimizer ##
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수 포함함
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam 사용하여 경사하강법

## Training & Back-propagation ##
total_batch = len(data_loader)
model.train()

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.view(-1, 32 * 32 * 3).to(device)
        Y = Y.to(device)

        optimizer.zero_grad() # 역전파 연산은 gradient 값을 누적시키므로 0으로 초기화
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward() # gradient 계산
        optimizer.step() # 새로 계산된 가중치를 업데이트하고 다음 에폭으로 넘어감

        avg_cost += cost / total_batch

    print('Epoch: ', '%04d' % (epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))
