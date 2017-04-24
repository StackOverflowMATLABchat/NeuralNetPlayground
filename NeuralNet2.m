classdef NeuralNet2 < handle
    % NeuralNet2 Neural Network implementation for the NeuralNetPlayground tool
    %   This class implements the training (forward and backpropagation) and
    %   predictions using Artificial Neural Networks (ANN).  The primary purpose
    %   is to assist in the construction of the NeuralNetPlayground framework
    %   as well as providing a framework for training neural networks that
    %   uses core MATLAB functionality only (i.e. no toolbox dependencies).
    %
    %   This class is initialized by specifying the total number of input
    %   layer neurons, hidden layer neurons for each hidden layer desired
    %   and the total number of output layer neurons as a single vector.
    %   With specified training examples and expected outputs, the neural
    %   network weights are learned with Stochastic Gradient Descent.
    %
    %   For regression, the number of output neurons is usually 1.  For
    %   binary class classification, the number of output neurons is usually 1
    %   and you perform the appropriate thresholding to decide which class
    %   an input belongs to.  For multi-class classification, the number of
    %   output neurons is usually the total number of classes in your data.
    %
    %   To do non-linear regression, you usually use a non-linear hidden
    %   activation function (sigmoid, tanh) with a linear output hidden
    %   activation function.  However, if you want to linear regression it may
    %   be more stable to specify no hidden layers and just have input
    %   and output layers only.  For classification, you can specify all layers
    %   to have the same activation function.  You also have the option of
    %   specifying no hidden layers and having a single input and output layer
    %   which will reduce to logistic regression.
    %
    %   The learned weights can be used to predict new examples given the
    %   learned weights.  The loss function used in this implementation is
    %   the sum of squared differences or Euclidean loss function.
    %
    % NeuralNet2 Properties:
    %   LearningRate       - The learning rate for Stochastic Gradient Descent
    %                        Must be strictly positive (i.e. > 0).
    %                        Default value is 0.03.
    %
    %   ActivationFunction - The activation function to be applied to each
    %                        neuron in the hidden and output layer.
    %                        The ones you can choose from are:
    %                        'linear': Linear
    %                        'relu': Rectified Linear Unit (i.e. ramp)
    %                        'tanh': Hyperbolic Tangent
    %                        'sigmoid': Sigmoidal
    %                        Default is 'tanh'.
    %
    %   OutputActivationFunction - The activation function to be applied
    %                              specifically to the output layer.  This will
    %                              change the output layer activation function
    %                              but leave the hidden layer one intact.
    %                              The ones you can choose are the same as
    %                              seen in the ActivationFunction property.
    %                              To set this to use the same activation function
    %                              as the hidden layer, set this to empty ([]).
    %                              Default is [].
    %
    %   RegularizationType - Apply regularization to the training process
    %                        if desired.
    %                        The ones you can choose from are:
    %                        'L1': L1 regularization
    %                        'L2': L2 regularization
    %                        'none': No regularization
    %                         Default is 'none'
    %                         Note: The method used for L1 regularization
    %                         comes from:
    %
    %                         Tsuroka, Y., Tsujii, J., Ananiadou, S.,
    %                         Stochastic Gradient Descent Training for
    %                         L1-regularized Log-linear Models with Cumulative
    %                         Penalty
    %                         http://aclweb.org/anthology/P/P09/P09-1054.pdf
    %
    %   RegularizationRate - The rate of regularization to apply (if desired)
    %                        Must be non-negative (i.e. >= 0).
    %                        Default is 0.
    %
    %   BatchSize          - Number of training examples selected per iteration
    %                        Choosing 1 example would implement true Stochastic
    %                        Gradient Descent while choosing the total number
    %                        of examples would implement Batch Gradient Descent.
    %                        Choosing any value in between implements mini-batch
    %                        Stochastic Gradient Descent.
    %                        Must be strictly positive (i.e. > 0) and integer.
    %                        Default is 10.  If the batch size is larger than
    %                        the total number of examples, the batch size
    %                        reverts to using the total number of examples.
    %
    %   ==========================
    %
    %   Example Use - Binary Classification
    %   -----------
    %   X = [0 0; 0 1; 1 0; 1 1]; % Define XOR data
    %   Y = [-1; 1; 1; -1];
    %   net = NeuralNet2([2 4 1]); % Create Neural Network object
    %                              % Two input layer neurons, one hidden
    %                              % layer with four neurons and one output
    %                              % layer neuron
    %   N = 5000;                  % Do 5000 iterations of Stochastic Gradient Descent
    %
    %   % Customize Neural Network engine
    %   net.LearningRate = 0.1;         % Learning rate is set to 0.1
    %   net.RegularizationType = 'L2';  % Regularization is L2
    %   net.RegularizationRate = 0.001; % Regularization rate is 0.001
    %
    %   perf = net.train(X, Y, N);  % Train the Neural Network
    %   Yraw = net.sim(X);          % Use trained object on original examples
    %   Ypred = ones(size(Yraw));   % Perform classification with thresholding
    %   Ypred(Yraw < 0) = -1;
    %   plot(1:N, perf);            % Plot cost function per iteration
    %
    %   % Display results
    %   disp('Training Examples and expected labels'); display(X); display(Y);
    %   disp('Predicted outputs'); display(Ypred);
    %   disp('Classification accuracy: ')
    %   disp(100 * sum(Y == Ypred) / size(X, 1));
    %
    %
    %   ==========================
    %
    %   Example Use - Multiclass Classification
    %   -----------
    %   load fisheriris; % Load the Fisher iris dataset
    %   numFeatures = size(meas, 2); % Total number of features - should be 4
    %   [~, ~, IDs] = unique(species); % Convert character labels to unique IDs
    %   numClasses = max(IDs); % Get total number of possible classes
    %   M = size(meas, 1); % Number of examples
    %   Y = full(sparse(1 : M, IDs.', 1, M, numClasses)); % Create an output
    %                                   % matrix where each row is an
    %                                   % example and each column denotes
    %                                   % the class the example belongs to.
    %                                   % That class gets assigned 1 while
    %                                   % the others get 0.  This should be
    %                                   % -1 if you are using tanh.
    %   net = NeuralNet2([numFeatures 4 numClasses]); % Create Neural Network object
    %                                   % Four input layer neurons, one hidden
    %                                   % layer with four neurons and three output layer
    %                                   % neuron
    %   N = 5000;                   % Do 5000 iterations of Stochastic Gradient Descent
    %   % Customize Neural Network engine
    %   net.LearningRate = 0.1;         % Learning rate is set to 0.1
    %   net.RegularizationType = 'L2';  % Regularization is L2
    %   net.RegularizationRate = 0.001; % Regularization rate is 0.001
    %   net.ActivationFunction = 'sigmoid'; % sigmoid hidden activation function
    %   perf = net.train(meas, Y, N);  % Train the Neural Network
    %   Yraw = net.sim(meas);         % Use trained object on original examples
    %   [~, Ypred] = max(Yraw, [], 2); % Determine which class has the largest
    %                                  % response per example
    %   plot(1:N, perf);            % Plot cost function per iteration
    %   % Display results
    %   disp('Training Examples and expected labels'); display(X); display(Y);
    %   disp('Predicted outputs'); display(Ypred);
    %   disp('Classification accuracy: ');
    %   disp(100 * sum(IDs == Ypred) / M);
    %
    %
    %   ==========================
    %
    %   Example Use - Regression
    %   -----------
    %   [x, y] = meshgrid(-2:0.5:2); % Define some sample two feature values
    %   z = x .* exp(-y); % Output is the function: x*(e^y)
    %   X = [x(:) y(:)]; % Unroll the features so they are each in a single column
    %   Y = z(:); % Unroll output of function into single column
    %   net = NeuralNet2([2 10 1]); % Create Neural Network object
    %                               % Two input layer neurons, one hidden
    %                               % layer with 10 neurons and one output layer
    %                               % neuron
    %   N = 10000;                  % Do 10000 iterations of Stochastic Gradient Descent
    %   % Customize Neural Network engine
    %   net.LearningRate = 0.1;         % Learning rate is set to 0.1
    %   net.RegularizationType = 'L2';  % Regularization is L2
    %   net.RegularizationRate = 0.001; % Regularization rate is 0.001
    %   net.ActivationFunction = 'Tanh'; % tanh hidden activation function
    %   net.OutputActivationFunction = 'linear';  % linear output activation function
    %   perf = net.train(X, Y, N);  % Train the Neural Network
    %   Ypred = net.sim(X);         % Use trained object on original examples
    %   plot(1:N, perf);            % Plot cost function per iteration
    %
    %   % Display results
    %   disp('Absolute difference between predicted and true values');
    %   disp(abs(Y - Ypred));
    %
    %   ==========================
    %
    %   See also NEURALNETAPP
    %
    %   StackOverflowMATLABchat - http://chat.stackoverflow.com/rooms/81987/matlab-and-octave
    %   Authors: Raymond Phan - http://stackoverflow.com/users/3250829/rayryeng
    %            Amro         - http://stackoverflow.com/users/97160/amro

    properties (Access = public)
        LearningRate % The learning rate (positive number)
        ActivationFunction % The desired activation function (string)
        OutputActivationFunction % The desired output activation function (string)
        RegularizationType % The type of regularization (string)
        RegularizationRate % The regularization rate (non-negative number)
        BatchSize % The size of the batch per iteration (positive integer number)
    end

    properties (Access = private)
        inputSize % Single value denoting how many neurons are in the input layer
        hiddenSizes % Vector denoting how many neurons per hidden layer
        outputSize % Single value denoting how many neurons are in the output layer
        weights % The weights of the neural network per layer
    end

    methods
        function this = NeuralNet2(layerSizes)
            % NeuralNet2  Create a Neural Network Instance
            %   The constructor takes in a vector of layer sizes where the
            %   first element denotes how many neurons are in the input layer,
            %   the next N elements denote how many neurons are in each desired
            %   hidden layer and the last element denotes how many neurons are
            %   in the output layer. Take note that the amount of neurons
            %   per layer that you specify does not include the bias units.
            %   These will be included when training the network. Therefore,
            %   the expected size of the vector is N + 2 where N is the total
            %   number of hidden layers for the neural network.  The exception
            %   to this rule is when you specify a vector of two elements.  This
            %   is interpreted as having an input layer, an output and no
            %   hidden layers.  This situation is when you would like to perform
            %   simple linear or logistic regression.
            %
            %   The following example creates a neural network with 1 input
            %   neuron (plus a bias) in the input layer, 2 hidden neurons
            %   (plus a bias) in the first hidden layer, 3 hidden neurons
            %   (plus a bias) in the second hidden layer and 1 output neuron
            %   in the output layer
            %
            %   layerSizes = [1 2 3 1];
            %   net = NeuralNet2(layerSizes);

            % default params
            this.LearningRate = 0.03;
            this.ActivationFunction = 'Tanh';
            this.OutputActivationFunction = [];
            this.RegularizationType = 'None';
            this.RegularizationRate = 0;
            this.BatchSize = 10;

            assert(numel(layerSizes) >= 2, 'Total number of layers should be at least 2');

            % network structure (fully-connected feed-forward)
            % Obtain input layer neuron size
            this.inputSize = layerSizes(1);

            % Obtain the hidden layer neuron sizes
            if numel(layerSizes ~= 2)
                this.hiddenSizes = layerSizes(2:end-1);
            else
                this.hiddenSizes = 0;
            end

            % Obtain the output layer neuron size
            this.outputSize = layerSizes(end);

            % Initialize matrices relating between the ith layer
            % and (i+1)th layer
            this.weights = cell(1, numel(layerSizes) - 1);
            for i = 1 : numel(layerSizes) - 1
                this.weights{i} = zeros(layerSizes(i) + 1, layerSizes(i + 1));
            end

            % Initialize weights
            init(this);
        end

        % ???
        function configure(this, X, Y)
            % check correct sizes
            [xrows,xcols] = size(X);
            [yrows,ycols] = size(Y);
            assert(xrows == yrows);
            assert(xcols == this.inputSize);
            assert(ycols == this.outputSize);

            % min/max of inputs/outputs
            inMin = min(X);
            inMax = max(X);
            outMin = min(Y);
            outMax = max(Y);
        end

        function init(this)
            % init  Initialize the Neural Network Weights
            %   This method initializes the neural network weights
            %   for connections using the initialization suggested by
            %   Kaiming He:
            %
            %   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
            %   Delving deep into rectifiers: Surpassing human-level performance
            %   on ImageNet classification. arXiv preprint arXiv:1502.01852.
            %   https://arxiv.org/abs/1502.01852
            %
            %   Take note that this method is run when you create an instance
            %   of the object. You would call init if you want to reinitialize
            %   the neural network and start from the beginning.
            %
            %   Uses:
            %       net = NeuralNet2([1 2 1]);
            %       % Other code...
            %       % ...
            %       % (Re-)initialize weights
            %       net.init();

            for ii = 1 : numel(this.weights)
                num = numel(this.weights{ii});
                % Kaiming He et al. initialization strategy
                this.weights{ii}(:) = 2.0 * randn(num, 1) / ...
                    sqrt(size(this.weights{ii}, 1));
            end
        end

        function perf = train(this, X, Y, numIter)
            % train  Perform neural network training with Stochastic Gradient Descent (SGD)
            %   This method performs training on the neural network structure
            %   that was specified when creating an instance of the class.
            %   Using training example features and their expected outcomes,
            %   trained network weights are created to facilitate future
            %   predictions.
            %
            %   Inputs:
            %      X - Training example features as a 2D matrix of size M x N
            %          M is the total number of examples and N are the
            %          total number of features.  M is expected to be the
            %          same size as the number of input layer neurons.
            %
            %      Y - Training example expected outputs as a 2D matrix of size
            %          M x P where M is the total number of examples and P is
            %          the total number of output neurons in the output layer.
            %
            %      numIter - Number of iterations Stochastic Gradient Descent
            %                should take while training. This is an optional
            %                parameter and the number of iterations defaults to
            %                1 if omitted.
            %
            %   Outputs:
            %      perf - An array of size numIter x 1 which denotes
            %             the cost between the predicted outputs and
            %             expected outputs at each iteration of learning
            %             the weights
            %
            %   Uses:
            %       net = NeuralNet2([1 2 1]); % Create NN object
            %       % Create example data here stored in X and Y
            %       %...
            %       %...
            %       perf = net.train(X, Y); % Perform 1 iteration
            %       perf2 = net.train(X, Y, 500); % Perform 500 iterations
            %
            %       See also NEURALNET2, SIM

            % If the number of iterations is not specified, assume 1
            if nargin < 4, numIter = 1; end

            % Ensure correct sizes
            assert(size(X, 1) == size(Y, 1), ['Total number of examples ' ...
                   'the inputs and outputs should match'])

            % Ensure regularization rate and batch size is proper
            assert(this.BatchSize >= 1, 'Batch size should be 1 or more');
            assert(this.RegularizationRate >= 0, ['Regularization rate ' ...
                   'should be 0 or larger']);

            % Check if we have specified the right regularization type
            regType = this.RegularizationType;
            assert(any(strcmpi(regType, {'l1', 'l2', 'none'})), ...
                   ['Ensure that you choose one of ''l1'', ''l2'' or '...
                   '''none'' for the regularization type']);

            % Ensure number of iterations is strictly positive
            assert(numIter >= 1, 'Number of iterations should be positive');

            % Initialize cost function array
            perf = zeros(1, numIter);

            % Total number of examples
            N = size(X, 1);

            % Total number of applicable layers
            L = numel(this.weights);

            % Get batch size
            % Remove decimal places in case of improper input
            B = floor(this.BatchSize);

            % Safely catch if batch size is larger than total number
            % of examples
            if B > N
                B = N;
            end

            % Cell array to store input and outputs of each neuron
            sNeuron = cell(1, L);

            % First cell array is for the initial
            xNeuron = cell(1, L + 1);

            % Cell array for storing the sensitivities
            delta = cell(1, L);

            % For L1 regularization
            if strcmpi(regType, 'l1')
                % This represents the total L1 penalty that each
                % weight could have received up to current point
                uk = 0;

                % Total penalty for each weight that was received up to
                % current point
                qk = cell(1, L);
                for ii = 1 : L
                    qk{ii} = zeros(size(this.weights{ii}));
                end
            end

            % Get activation function for the hidden layer
            fcn = getActivationFunction(this.ActivationFunction);

            % Get derivative of activation function for the hidden layer
            dfcn = getDerivativeActivationFunction(this.ActivationFunction);

            % Do the same for the output layer
            if isempty(this.OutputActivationFunction)
                fcnOut = fcn;
                dfcnOut = dfcn;
            else
                fcnOut = getActivationFunction(this.OutputActivationFunction);
                dfcnOut = getDerivativeActivationFunction(this.OutputActivationFunction);
            end

            % For each iteration...
            for ii = 1 : numIter
                % If the batch size is equal to the total number of examples
                % don't bother with random selection as this will be a full
                % batch gradient descent
                if N == B
                    ind = 1 : N;
                else
                    % Randomly select examples corresponding to the batch size
                    % if the batch size is not equal to the number of examples
                    ind = randperm(N);
                    ind = ind(1 : B);
                end

                % Select out the training example features and expected outputs
                IN = X(ind, :);
                OUT = Y(ind, :);

                % Initialize input layer
                xNeuron{1} = [IN ones(B, 1)];

                %%% Perform forward propagation
                % Make sure you save the inputs and outputs into each neuron
                % at the hidden and output layers
                for jj = 1 : L
                    % Compute inputs into next layer
                    sNeuron{jj} = xNeuron{jj} * this.weights{jj};

                    % Compute outputs of this layer
                    if jj == L
                        xNeuron{jj + 1} = fcnOut(sNeuron{jj});
                    else
                        xNeuron{jj + 1} = [fcn(sNeuron{jj}) ones(B, 1)];
                    end
                end

                %%% Perform backpropagation

                % Compute sensitivities for output layer
                delta{end} = (xNeuron{end} - OUT) .* dfcnOut(sNeuron{end});

                % Compute the sensitivities for the rest of the layers
                for jj = L - 1 : -1 : 1
                    delta{jj} = dfcn(sNeuron{jj}) .* ...
                        (delta{jj + 1}*(this.weights{jj + 1}(1 : end - 1, :)).');
                end

                %%% Compute weight updates
                alpha = this.LearningRate;
                lambda = this.RegularizationRate;
                for jj = 1 : L
                    % Obtain the outputs and sensitivities for each
                    % affected layer
                    XX = xNeuron{jj};
                    D = delta{jj};

                    % Calculate batch weight update
                    weight_update = (1 / B) * (XX.') * D;

                    % Apply L2 regularization if required
                    if strcmpi(regType, 'l2')
                        weight_update(1 : end-1, :) = weight_update(1 : end - 1, :) + ...
                            (lambda/B)*this.weights{jj}(1 : end - 1, :);
                    end

                    % Compute the final update
                    this.weights{jj} = this.weights{jj} - alpha * weight_update;
                end

                % Apply L1 regularization if required
                if strcmpi(regType, 'l1')
                    % Step #1 - Accumulate total L1 penalty that each
                    % weight could have received up to this point
                    uk = uk + (alpha * lambda / B);

                    % Step #2
                    % Using the updated weights, now apply the penalties
                    for jj = 1 : L
                        % 2a - Save previous weights and penalties
                        % Make sure to remove bias terms
                        z = this.weights{jj}(1 : end - 1,:);
                        q = qk{jj}(1 : end - 1,:);

                        % 2b - Using the previous weights, find the weights
                        % that are positive and negative
                        w = z;
                        indwp = w > 0;
                        indwn = w < 0;

                        % 2c - Perform the update on each condition
                        % individually
                        w(indwp) = max(0, w(indwp) - (uk + q(indwp)));
                        w(indwn) = min(0, w(indwn) + (uk - q(indwn)));

                        % 2d - Update the actual penalties
                        qk{jj}(1:end-1,:) = q + (w - z);

                        % Don't forget to update the actual weights!
                        this.weights{jj}(1 : end - 1, :) = w;
                    end
                end

                % Compute cost at this iteration
                perf(ii) = (0.5 / B)*sum(sum((xNeuron{end} - OUT).^2));

                % Add in regularization if necessary
                if strcmpi(regType, 'l1')
                    for jj = 1 : L
                        perf(ii) = perf(ii) + ...
                            (lambda / B) * sum(sum(abs(this.weights{jj}(1 : end - 1, :))));
                    end
                elseif strcmpi(regType, 'l2')
                    for jj = 1 : L
                        perf(ii) = perf(ii) + ...
                            (0.5 * lambda / B) * sum(sum((this.weights{jj}(1 : end - 1, :)).^2));
                    end
                end
            end
        end

        function [OUT, OUTS] = sim(this, X)
            % sim  Perform Neural Network Predictions
            %   This method performs forward propagation using the
            %   learned weights after training. Forward propagation
            %   uses the learned weights to propogate information
            %   throughout the neural network and the predicted outcome
            %   is seen at the output layer.
            %
            %   Inputs:
            %      X - Training examples to predict their outcomes
            %          This is a M x N matrix where M is the total number of
            %          examples and N is the total number of features.
            %          N must equal to the total number of neurons in the input
            %          layer
            %
            %   Outputs:
            %      OUT - The predicted outputs using the training examples in X.
            %            This is a M x P matrix where M is the total number of
            %            training examples and P is the total number of output
            %            neurons
            %
            %     OUTS - This is a 1D cell array of size 1 x NN where NN
            %            is the total number of layers in the neural network
            %            including the input and output layers.  Therefore, if
            %            K is equal to the total number of hidden layers,
            %            NN = K+2.  This cell array contains the outputs of
            %            each example per layer. Specifically, each element
            %            OUTS{ii} would be a M x Q matrix where Q would be the
            %            total number of neurons in layer ii without the bias
            %            unit. ii=1 is the input layer and ii=NN is the output
            %            layer.  Remember if you specify a network with no hidden
            %            layers, this would mean that K=0 and so NN = 2.
            %            OUTS{ii} would contain all of the outputs for each
            %            example in layer ii.
            %
            %   Uses:
            %       net = NeuralNet2([1 2 1]); % Create NN object
            %       % Create your training data and train your neural network
            %       % here...
            %       % Also create test data here stored in XX...
            %       % ...
            %       OUT = net.sim(XX); % Find predictions
            %       [OUT,OUTS] = net.sim(XX); % Find predictions and outputs
            %                                 % per layer
            %
            %   See also NEURALNET2, TRAIN

            % Check if the total number of features matches the
            % total number of input neurons
            assert(size(X, 2) == this.inputSize, ['Number of features '...
                   'should match the number of input neurons']);

            % Get total number of examples
            N = size(X, 1);

            %%% Begin algorithm
            % Start with input layer
            OUT = X;

            % Also initialize cell array with input layer's contents
            OUTS = cell(1, numel(this.weights) + 1);
            OUTS{1} = OUT;

            % Get activation function
            fcn = getActivationFunction(this.ActivationFunction);

            if isempty(this.OutputActivationFunction)
                fcnOut = fcn;
            else
                fcnOut = getActivationFunction(this.OutputActivationFunction);
            end

            % For each layer...
            for ii = 1 : numel(this.weights)
                % Compute inputs into each neuron and corresponding
                % outputs
                if ii == numel(this.weights)
                    OUT = fcnOut([OUT ones(N, 1)] * this.weights{ii});
                else
                    OUT = fcn([OUT ones(N, 1)] * this.weights{ii});
                end
                OUTS{ii + 1} = OUT;
            end
        end
    end
end

function fcn = getActivationFunction(activation)
    switch lower(activation)
        case 'linear'
            fcn = @f_linear;
        case 'relu'
            fcn = @f_relu;
        case 'tanh'
            fcn = @f_tanh;
        case 'sigmoid'
            fcn = @f_sigmoid;
        otherwise
            error('Unknown activation function');
    end
end

function fcn = getDerivativeActivationFunction(activation)
    switch lower(activation)
        case 'linear'
            fcn = @fd_linear;
        case 'relu'
            fcn = @fd_relu;
        case 'tanh'
            fcn = @fd_tanh;
        case 'sigmoid'
            fcn = @fd_sigmoid;
        otherwise
            error('Unknown activation function');
    end
end

% activation funtions and their derivatives
function y = f_linear(x)
    % See also: purelin
    y = x;
end

function y = fd_linear(x)
    % See also: dpurelin
    y = ones(size(x));
end

function y  = f_relu(x)
    % See also: poslin
    y = max(x, 0);
end

function y = fd_relu(x)
    % See also: dposlin
    y = double(x >= 0);
end

function y = f_tanh(x)
    % See also: tansig
    %y = 2 ./ (1 + exp(-2*x)) - 1;
    y = tanh(x);
end

function y = fd_tanh(x)
    % See also: dtansig
    y = f_tanh(x);
    y = 1 - y.^2;
end

function y = f_sigmoid(x)
    % See also: logsig
    y = 1 ./ (1 + exp(-x));
end

function y = fd_sigmoid(x)
    % See also: dlogsig
    y = f_sigmoid(x);
    y = y .* (1 - y);
end
