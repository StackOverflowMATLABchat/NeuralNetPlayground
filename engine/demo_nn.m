%%% Neural Network Engine Example
clearvars;
close all;

%%% 1. Declare total number of input neurons, hidden layer neurons and output
%%% neurons
input_neurons = 2;
hidden_neurons = [4 2];
output_neurons = 1;

%%% 2. Set total number of iterations
N = 10000;

%%% 3. Create synthetic dataset
rng(123);
M = 100;
tol = 0.1;
x1 = bsxfun(@plus, rand(M,2) - 0.5, [-0.5-tol, 0.5+tol]); % Top-Left quadrant
x2 = bsxfun(@plus, rand(M,2) - 0.5, [0.5+tol, 0.5+tol]); % Top-Right quadrant
x3 = bsxfun(@plus, rand(M,2) - 0.5, [-0.5-tol, -0.5-tol]); % Bottom-Left quadrant
x4 = bsxfun(@plus, rand(M,2) - 0.5, [0.5+tol, -0.5-tol]); % Bottom-Right quadrant

% Create final dataset
X = [x1; x2; x3; x4];
Y = [-ones(M,1); ones(2*M,1); -ones(M,1)];

% Randomly sample and create training and test sets
per = 0.5;
m = numel(Y);
ind = randperm(m);
pt = floor(per*m);
Xtrain = X(ind(1:pt), :);
Ytrain = Y(ind(1:pt));
Xtest = X(ind(pt+1:end),:);
Ytest = Y(ind(pt+1:end));
mTrain = size(Xtrain,1);

%%% 4. Declare NN engine
NN = NeuralNet2([input_neurons, hidden_neurons, output_neurons]);
NN.LearningRate = 0.1;
NN.RegularizationType = 'L2';
NN.RegularizationRate = 0.01;
NN.ActivationFunction = 'Tanh';
NN.BatchSize = 10;
NN.Debug = true;

%%% 6. Find optimal weights
costVal = NN.train(Xtrain, Ytrain, N);
figure;
gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], ...
    'x');
hold on;
gscatter(Xtest(:,1), Xtest(:,2), Ytest, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], ...
    '.');

%%% 7. Compute predictions for training and testing data
predicthTrain = 2*double(NN.sim(Xtrain) >= 0) - 1;
predicthTest = 2*double(NN.sim(Xtest) >= 0) - 1;

%%% 8. Compute classification accuracy for training and testing data
fprintf('Classification Accuracy for Training Data: %f\n', ...
        100*sum(predicthTrain == Ytrain) / numel(Ytrain));

fprintf('Classification Accuracy for Testing Data: %f\n', ...
        100*sum(predicthTest == Ytest) / numel(Ytest));
    
fprintf('The final cost to assign the optimal weights are: %f\n', costVal(end));

%%% 9. Plotting of cost function per epoch
figure;
plot(1:N, costVal);
title('Cost function vs. epoch');
xlabel('Epoch');
ylabel('Cost Function');
grid;

%%% 10. Plot decision regions
[X,Y] = meshgrid(linspace(-1.25,1.25,1000));
Xout = [X(:) Y(:)];
predictDes = 2*double(NN.sim(Xout) >= 0) - 1;
figure;

% Plot decision regions first
h = plot(Xout(predictDes==-1,1), Xout(predictDes==-1,2), '.', 'MarkerEdgeColor', ...
    [0.9290;0.6940;0.1250]);
hold on;
h2 = plot(Xout(predictDes==1,1), Xout(predictDes==1,2), '.', 'MarkerEdgeColor', ...
    [0.4940;0.1840;0.5560]);

% Now plot points
plot(Xtrain(Ytrain==-1,1), Xtrain(Ytrain==-1,2), '.', 'MarkerEdgeColor', ...
    [0 0.4470 0.7410], 'MarkerSize', 12);
plot(Xtrain(Ytrain==1,1), Xtrain(Ytrain==1,2), '.', 'MarkerEdgeColor', ...
    [0.8500 0.3250 0.0980], 'MarkerSize', 12);
plot(Xtest(Ytest==-1,1), Xtest(Ytest==-1,2), 'x', 'MarkerEdgeColor', ...
    [0 0.4470 0.7410], 'MarkerSize', 12);
plot(Xtest(Ytest==1,1), Xtest(Ytest==1,2), 'x', 'MarkerEdgeColor', ...
    [0.8500 0.3250 0.0980], 'MarkerSize', 12);
axis tight;
legend('Decision Region - Negative', 'Decision Region - Positive', ...
    'Training Examples - Negative', 'Training Examples - Positive', ...
    'Test Examples - Negative', 'Test Examples - Positive');
