'''

https://rubber-tree.tistory.com/139

'''

import torch
import torchvision.datasets as dsets
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
batch_size = 100 # 연산 한 번에 들어가는 데이터 크기
drop_prob = 0.3 # dropout 확률

## Dataset 설정 ##
mnist_train = dsets.MNIST(root='MNIST-data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST-data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

## 모델 설계 ##
# MLP이므로 여러 Layer 설정
# mnist는 28 * 28(784) 픽셀 크기의 데이터
# 보통 512로 한다 함
# 최종 결과는 0~9의 정수값
linear1 = nn.Linear(784, 512, bias=True) # nn.Linear(input dem, output dem)
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
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad() # 역전파 연산은 gradient 값을 누적시키므로 0으로 초기화
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward() # gradient 계산
        optimizer.step() # 새로 계산된 가중치를 업데이트하고 다음 에폭으로 넘어감

        avg_cost += cost / total_batch

    print('Epoch: ', '%04d' % (epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))

## 검증 ##
with torch.no_grad(): # gradient 계산을 수행 안 함
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()