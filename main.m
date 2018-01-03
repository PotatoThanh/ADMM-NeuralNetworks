clear all, close all, clc;
g = gpuDevice(1);
reset(g);

%load data
% load data
load trainX
load trainY
load testY
load testX

n_hidden_1 = 256; % 1st layer number of features
n_hidden_2 = 256; % 2nd layer number of features
n_input = 784; % MNIST data input (img shape: 28*28)
n_classes = 10; % MNIST total classes (0-9 digits) 
n_batchsize = size(trainX, 2); %MNIST number of sample

nn.a0Size = [n_input, n_batchsize]; %784xn

nn.w1Size = [n_hidden_1, n_input]; %256x784
nn.w2Size = [n_hidden_2, n_hidden_1]; %256x256
nn.w3Size = [n_classes, n_hidden_2]; %10x256

nn.z1Size = [n_hidden_1, n_batchsize]; %256xn
nn.a1Size = [n_hidden_1, n_batchsize]; %256xn
nn.z2Size = [n_hidden_2, n_batchsize]; %256xn
nn.a2Size = [n_hidden_2, n_batchsize]; %256xn
nn.z3Size = [n_classes , n_batchsize]; %10xn

nn.lambda = [n_classes, n_batchsize]; %10xn

nn.beta3  = 5;
nn.beta2  = 5;
nn.beta1  = 5;

nn.gama2  = 5;
nn.gama1  = 5;

nn.rho    = 0.0;

nn.maxIter = 50;
nn.numLayer = 3;

%running on CPU
%gADMM_NN(dataX, dataY, nn)

%init variable
a0 = gpuArray.zeros(nn.a0Size);

w1 = gpuArray.zeros(nn.w1Size); %256x784
w2 = gpuArray.zeros(nn.w2Size); %256x256
w3 = gpuArray.zeros(nn.w3Size); %10x256

z1 = gpuArray.randn(nn.z1Size); %256xn
a1 = gpuArray.randn(nn.a1Size); %256xn
z2 = gpuArray.randn(nn.z2Size); %256xn
a2 = gpuArray.randn(nn.a2Size); %256xn
z3 = gpuArray.randn(nn.z3Size); %10xn

lambda = gpuArray.ones(nn.lambda); %10xn

beta3  = gpuArray(nn.beta3);
beta2  = gpuArray(nn.beta2);
beta1  = gpuArray(nn.beta1);

gama2  = gpuArray(nn.gama2);
gama1  = gpuArray(nn.gama1);

rho    = gpuArray(nn.rho);

maxIter  = gpuArray(nn.maxIter);
numLayer = gpuArray(nn.numLayer);

% assign data
trainX = gpuArray(trainX);
trainY = gpuArray(trainY);
testX  = gpuArray(testX);
testY  = gpuArray(testY);
a0 = trainX;
y_labels  = trainY;

%warm start
warm = 1;
if warm
    for i=1:5
        disp('--warming--');
        disp(i);        
        
        %layer 1
        w1  =  weight_update(z1, a0, rho);
        a1  =  activation_update(w2, z2, z1, beta2, gama1);
        z1  =  argminz(a1, w1, a0, beta1, gama1);

        %layer 2
        w2  =  weight_update(z2, a1, rho);
        a2  =  activation_update(w3, z3, z2, beta3, gama2);
        z2  =  argminz(a2, w2, a1, beta2, gama2);

        %layer 3
        w3  =  weight_update(z3, a2, rho);
        z3  =  argminlastz(y_labels, lambda, w3, a2, beta3);

    end
end

iter_TrainLoss = [];
iter_TestLoss = [];
iter_TrainAccuracy = [];
iter_TestAccuracy = [];
for i=1:maxIter
    disp('----------------------------');
    disp('--training--');
    disp(i);    
    %layer 1
    w1  =  weight_update(z1, a0, rho);
    a1  =  activation_update(w2, z2, z1, beta2, gama1);
    z1  =  argminz(a1, w1, a0, beta1, gama1);

    %layer 2
    w2  =  weight_update(z2, a1, rho);
    a2  =  activation_update(w3, z3, z2, beta3, gama2);
    z2  =  argminz(a2, w2, a1, beta2, gama2);

    %layer 3
    w3  =  weight_update(z3, a2, rho);
    z3  =  argminlastz(y_labels, lambda, w3, a2, beta3);
    lambda = lambda_update(z3, w3, a2, beta3);
    

    % Training data
    forward = w3*relu(w2*relu(w1*trainX));
    loss_train = (forward - trainY).^2;
    loss_train = sum(loss_train(:))
    [M1, I1] = max(y_labels);
    [M2, I2] = max(forward);
    accracy_train = mean(I1 == I2)

    disp('----');

    %test data
    forward1 = w3*relu(w2*relu(w1*testX));
    loss_test = (forward1 - testY).^2;
    loss_test = sum(loss_test(:))
    [M11, I11] = max(testY);
    [M22, I22] = max(forward1);
    accracy_test = mean(I11 == I22)
    
    %drawing
    iter_TrainLoss = [iter_TrainLoss, loss_train];
    iter_TrainAccuracy = [iter_TrainAccuracy, accracy_train];
    iter_TestLoss = [iter_TestLoss, loss_test];
    iter_TestAccuracy = [iter_TestAccuracy, accracy_test];

    figure(1);
    hold on;
    plot(iter_TrainLoss);
    hold on;
    plot(iter_TestLoss);
    xlabel('Iterations');
    ylabel('loss');
    drawnow();

    figure(2);
    hold on;
    plot(iter_TrainAccuracy);
    hold on;
    plot(iter_TestAccuracy);
    xlabel('Iterations');
    ylabel('accuracy');
    drawnow();
end