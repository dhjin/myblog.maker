---
title: Multi Layer Perceltron
---
출처: https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl


```julia
using Flux,Plots
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using MLDatasets
using CUDAapi
```

### Flux 패키지 및 함수
- Flux.Data 에서 DataLoader: 데이로 입출력을 관할
- onehotbatch: 정수를 원핫벡터로 변환
- onecold: 0-1 반전.
- MLDatasets: 머신러닝을 위한 오픈소스 데이터세트
- CUDAapi: GPU 사용을 위한 패키지

## [1] CUDA 세팅


```julia
if has_cuda() # CUDA 패키지를 활용 GPU 사용 여부를 감지한다.
    @info "CUDA is on"
    import CuArrays # 가능하다면 CuArrays를 import해 온다.
    CuArrays.allowscalar(false)
end
```

## [2] 파라미터 세팅
- @with_kw를 사용, 파라미터 값들을 Args 변수에 할당함.
- mutable struct Args: "Args"라는 새로운 자료유형 설정, 안의 필드 값들이 변할 수 도 있음을 의미함
- η 라는 실수 유형의 변수: learning rate 설정
- 배치 사이즈, 반복횟수, 계산장비


```julia
@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end
```




    Args



## [3] 데이터 가져오기 및 전처리
- [1] 손글씨 DB 가져오기
- [2] 정답레이블(y) 원핫벡터 만들기
- [3] 학습/테스트 데이터 분할하여 DataLoader안에 넣기

모든 과정이 getdata함수에 포함


```julia
#[1] 학습세트: 28x28x60000 실수 벡터 
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
```




    (Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    ...
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4  …  9, 2, 9, 5, 1, 8, 3, 5, 6, 8])




```julia
# [2]
ytrain = onehotbatch(ytrain, 0:9)
```




    10×60000 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
     0  1  0  0  0  0  0  0  0  0  0  0  0  …  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  1  0  0  1  0  1  0  0  0  0     0  0  0  0  0  0  1  0  0  0  0  0
     0  0  0  0  0  1  0  0  0  0  0  0  0     0  0  0  1  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  1  0  0  1  0  1     0  0  0  0  0  0  0  0  1  0  0  0
     0  0  1  0  0  0  0  0  0  1  0  0  0     0  0  0  0  0  0  0  0  0  0  0  0
     1  0  0  0  0  0  0  0  0  0  0  1  0  …  0  0  0  0  0  1  0  0  0  1  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0     0  0  0  0  0  0  0  0  0  0  1  0
     0  0  0  0  0  0  0  0  0  0  0  0  0     1  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0     0  1  0  0  0  0  0  1  0  0  0  1
     0  0  0  0  1  0  0  0  0  0  0  0  0     0  0  1  0  1  0  0  0  0  0  0  0




```julia
# [3]
args = Args()
train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
```




    DataLoader((Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    ...
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
    
    Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], Bool[0 1 … 0 0; 0 0 … 0 0; … ; 0 0 … 0 1; 0 0 … 0 0]), 1024, 60000, true, 60000, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  59991, 59992, 59993, 59994, 59995, 59996, 59997, 59998, 59999, 60000], true)




```julia
function getdata(args)
    # Loading Dataset
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return train_data, test_data
end
```




    getdata (generic function with 1 method)



## [4] 모델 만들기(layer쌓기)
- 입력층: 28x28입력, 퍼셉트론 개수 32개, 활성함수 relu
- 출력층: 입력 32, 출력 10\
참고자료: https://fluxml.ai/Flux.jl/stable/models/basics/#Building-Layers-1


```julia
function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
        Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end
```




    build_model (generic function with 1 method)



## [5] 손실 함수


```julia
# 
function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end
```




    loss_all (generic function with 1 method)




```julia

```


```julia
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end
```




    accuracy (generic function with 1 method)



## multi-layer perceptron 전체 학습

- wrapper @epochs 는 반복횟수를 지정해서 훈련함수를 실행하도록 한다.
- Flux.train 함수가 실제 훈련을 수행 필요 파라미터는 아래와 같다.
> loss\
params\
train_data\
opt\
cb


```julia
# 훈련시키기
function train(; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    # 로스 함수.
    loss(x,y) = logitcrossentropy(m(x), y)
    
    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)
    
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)

    @show accuracy(test_data, m)
end
```




    train (generic function with 1 method)




```julia

```
