classdef NeuralNetApp < handle
    %NEURALNETAPP  Neural network Application
    %
    % Inspired by TensoreFlow playground:
    % http://playground.tensorflow.org/
    %
    % See also: NeuralNet2
    %

    %% Properties
    properties (SetAccess = private)
        % structure containing graphics handles
        handles
        % structure containing data
        data
        % neural network object
        net
    end

    properties (Access = private)
        % app state (running or paused)
        isRunning = false;
    end

    properties (Access = private, Constant = true)
        % maximuum number of hidden layers in the UI
        MAX_LAYERS = 5;
        % maximum number of neurons in each hidden layer in the UI
        MAX_NEURONS = 8;
        % data domain for both X/Y dimensions
        DOM = [-6 6];
        % colormap
        CMAP = getColormap();
    end

    %% Constructor
    methods
        function this = NeuralNetApp()
            %NEURALNETAPP  Constructor

            % initialize UI
            createGUI(this);
            updateHiddenLayers(this, [4 2]);

            % generate data
            genData(this);
            updatePlotScatter(this);

            % create network
            buildNet(this);
            reset(this);
        end

        function delete(this)
            %DELETE  Destructor

            % close figure
            %delete(this.handles.hFig);
        end
    end

    %% Private Methods
    methods (Access = private)
        function genData(this)
            %GENDATA  Generate new data according to selected dataset and options

            % generate 2D points with corresponding binary labels {-1,+1}
            N = 500;  % number of points
            ratio = get(this.handles.hSlidRatio, 'Value') / 100;
            noise = get(this.handles.hSlidNoise, 'Value') / 100;
            switch get(this.handles.hPopData, 'Value')
                case 1
                    [points,labels] = genDataCircle(N, noise);
                case 2
                    [points,labels] = genDataXOR(N, noise);
                case 3
                    [points,labels] = genDataGaussian(N, noise);
                case 4
                    [points,labels] = genDataSpiral(N, noise);
                otherwise
                    error('Unrecognized dataset');
            end

            % clip to [-6,6] range
            %points = min(max(points, this.DOM(1)), this.DOM(2));

            % store data, partition indices, and mapped features
            this.data.points = points;
            this.data.labels = labels;
            this.data.trainIDX = splitData(labels, ratio);
            this.data.inputs = mapPoints(points, getInputsMask(this));
        end

        function buildNet(this)
            %BUILDNET  Create new neural network object

            % build net using specified layer sizes
            inputSize = size(this.data.inputs, 2);
            hiddenSizes = getHiddenSizes(this);
            outputSize = 1;
            this.net = NeuralNet2([inputSize hiddenSizes outputSize]);

            % set network parameters from GUI options
            this.net.BatchSize = round(get(this.handles.hSlidBatch, 'Value'));

            vals = cellstr(get(this.handles.hPopLearnRate, 'String'));
            idx = get(this.handles.hPopLearnRate, 'Value');
            this.net.LearningRate = str2double(vals{idx});

            vals = cellstr(get(this.handles.hPopActFunc, 'String'));
            idx = get(this.handles.hPopActFunc, 'Value');
            this.net.ActivationFunction = vals{idx};

            vals = cellstr(get(this.handles.hPopRegType, 'String'));
            idx = get(this.handles.hPopRegType, 'Value');
            this.net.RegularizationType = vals{idx};

            vals = cellstr(get(this.handles.hPopRegRate, 'String'));
            idx = get(this.handles.hPopRegRate, 'Value');
            this.net.RegularizationRate = str2double(vals{idx});

            vals = cellstr(get(this.handles.hPopProblem, 'String'));
            idx = get(this.handles.hPopProblem, 'Value');
            switch vals{idx}
                case 'Classification'
                case 'Regression'
            end

            % configure network
            configure(this.net, this.data.inputs(this.data.trainIDX,:), ...
                this.data.labels(this.data.trainIDX));
        end

        function reset(this)
            %RESET  Reset UI state

            % paused, reset iterations, reset loss plots
            this.isRunning = false;
            set(this.handles.hBtnRun, 'String','Run');
            set(this.handles.hTxtIter, 'String','000');
            set(this.handles.hLineLoss, 'XData',NaN, 'YData',NaN);
            set(this.handles.hLgndLoss, 'String',...
                {'Test loss: 0.000', 'Training loss: 0.000'});

            % re-initialize network, and run a pass to update UI
            init(this.net);
            step(this, true);
            drawnow();
        end

        function step(this, skipTrain)
            %STEP  Run one iteration: train, evaluate, and update UI

            if nargin < 2, skipTrain = false; end

            % data
            inputTrain = this.data.inputs(this.data.trainIDX,:);
            inputTest = this.data.inputs(~this.data.trainIDX,:);
            labelsTrain = this.data.labels(this.data.trainIDX);
            labelsTest = this.data.labels(~this.data.trainIDX);

            if ~skipTrain
                % increment iteration count
                iter = str2double(get(this.handles.hTxtIter, 'String'));
                set(this.handles.hTxtIter, 'String',sprintf('%03d',iter+1));

                % train network
                train(this.net, inputTrain, labelsTrain);
            end

            % evaluate network
            lossTrain = mseLoss(sim(this.net, inputTrain), labelsTrain);
            lossTest = mseLoss(sim(this.net, inputTest), labelsTest);

            % update plots
            updatePlotLoss(this, lossTest, lossTrain);
            updatePlotHeatmaps(this);
        end

        function updatePlotLoss(this, lossTest, lossTrain)
            %UPDATEPLOTLOSS  Update loss plot

            % add new point to each line
            y1 = [get(this.handles.hLineLoss(1), 'YData'), lossTest];
            y2 = [get(this.handles.hLineLoss(2), 'YData'), lossTrain];
            x = [NaN 2:numel(y1)];
            set(this.handles.hLineLoss(1), 'XData',x, 'YData',y1);
            set(this.handles.hLineLoss(2), 'XData',x, 'YData',y2);

            % update legend strings
            set(this.handles.hLgndLoss, 'String',...
                {sprintf('Test loss: %.3f',lossTest), ...
                sprintf('Training loss: %.3f',lossTrain)});
        end

        function updatePlotHeatmaps(this)
            %UPDATEPLOTHEATMAPS  Update heatmap plots (hidden and final output)

            % options
            inputsIdx = getInputsMask(this);
            discretize = logical(get(this.handles.hCBoxDiscretize, 'Value'));

            % update hidden neurons heatmaps
            [X1,X2] = meshgrid(linspace(this.DOM(1),this.DOM(2),30));
            [~,A] = sim(this.net, mapPoints([X1(:) X2(:)], inputsIdx));
            A = A(2:end-1);  % ignore input/output layers outputs
            for layer=1:numel(A)
                a = reshape(A{layer}, 30, 30, []);
                for neuron=1:size(a,3)
                    img = genThumbnail(a(:,:,neuron), this.CMAP, discretize);
                    set(this.handles.hBtnNeuron(neuron,layer), 'CData',img);
                end
            end

            % update output heatmap
            [X1,X2] = meshgrid(linspace(this.DOM(1),this.DOM(2),250));
            a = sim(this.net, mapPoints([X1(:) X2(:)], inputsIdx));
            a = reshape(a, 250, 250);
            if discretize
                a = sign(a);
            end
            set(this.handles.hImgOut, 'CData',a);
        end

        function updatePlotScatter(this)
            %UPDATEPLOTSCATTER  Update scatter plots

            % data
            ptsTrain = this.data.points(this.data.trainIDX,:);
            ptsTest = this.data.points(~this.data.trainIDX,:);
            labelsTrain = this.data.labels(this.data.trainIDX);
            labelsTest = this.data.labels(~this.data.trainIDX);

            % update scatter points
            klass = [-1 1];
            for k=1:numel(klass)
                idx = (labelsTrain == klass(k));
                set(this.handles.hLineTrain(k), ...
                    'XData',ptsTrain(idx,1), 'YData',ptsTrain(idx,2));
                idx = (labelsTest == klass(k));
                set(this.handles.hLineTest(k), ...
                    'XData',ptsTest(idx,1), 'YData',ptsTest(idx,2));
            end
        end

        function updateHiddenLayers(this, hiddenSizes)
            %UPDATEHIDDENLAYERS  Update hidden layers UI to match new size

            % mask of active layers and neurons
            maskL = ((1:this.MAX_LAYERS) <= numel(hiddenSizes));
            maskN = zeros(1, this.MAX_LAYERS);
            maskN(maskL) = hiddenSizes;
            maskN = bsxfun(@le, (1:this.MAX_NEURONS)', maskN);

            % refresh layers count and add/remove buttons
            set(this.handles.hTxtLayerNum, ...
                'String',pluralize(numel(hiddenSizes), 'hidden layer'));
            [sAdd, sDel] = getButtonState(numel(hiddenSizes), this.MAX_LAYERS);
            set(this.handles.hBtnLayerAdd, 'Enable',sAdd);
            set(this.handles.hBtnLayerDel, 'Enable',sDel);

            % refresh neurons heatmaps
            vals = {'off'; 'on'};
            set(this.handles.hBtnNeuron(:), {'Visible'},vals(maskN(:)+1));

            % refresh neurons count
            set(this.handles.hTxtNeuronNum(~maskL), 'String','0 neuron');
            set(this.handles.hTxtNeuronNum(maskL), {'String'},...
                cellstr(pluralize(hiddenSizes(:), 'neuron')));

            % refresh neurons add/remove buttons
            set([this.handles.hBtnNeuronAdd(~maskL), ...
                this.handles.hBtnNeuronDel(~maskL)], 'Enable','off');
            for i=1:numel(hiddenSizes)
                [sAdd, sDel] = getButtonState(hiddenSizes(i), this.MAX_NEURONS);
                set(this.handles.hBtnNeuronAdd(i), 'Enable',sAdd);
                set(this.handles.hBtnNeuronDel(i), 'Enable',sDel);
            end
        end

        function idx = getInputsMask(this)
            %GETINPTUSMASK  Return a mask of selected inputs from UI

            val = get(this.handles.hTBtnInput, 'Value');
            idx = logical(cell2mat(val));
        end

        function sz = getHiddenSizes(this)
            %GETHIDDENSIZES  Return hidden layers sizes from UI

            str = get(this.handles.hTxtNeuronNum, 'String');
            sz = nonzeros(cellfun(@(s) sscanf(s, '%d'), str)).';
        end
    end

    %% UI
    methods (Access = private)
        function createGUI(this)
            %CREATEGUI  Build the UI

            % main figure
            hFig = figure('Menubar','none', 'Toolbar','none', ...
                'NumberTitle','off', 'Name','Neural Network Playground', ...
                'Colormap',this.CMAP, 'Resize','off', 'Visible','off', ...
                'Units','pixels', 'Position',[100 100 1000 600]);

            this.handles = struct();
            this.handles.hFig = hFig;

            % create panels
            hPan = createGUI_Panels(this, hFig);
            createGUI_PanelHeader(this, hPan(1));
            createGUI_PanelData(this, hPan(2));
            createGUI_PanelInput(this, hPan(3));
            createGUI_PanelLayers(this, hPan(4));
            createGUI_PanelOutput(this, hPan(5));

            % setup event handlers
            registerCallbacks(this);

            % make figure visible
            set(hFig, 'Visible','on');
        end

        function hPan = createGUI_Panels(this, hParent)
            %CREATEGUI_PANELS  Create the layout of the top-level panels

            % panel properties
            props = {'ForegroundColor',0.3*[1 1 1]};
            titles = upper({'', 'Data', 'Input', 'Hidden Layers', 'Output'});
            pos = {[5 520 990 75], [5 5 135 510], [145 5 115 510], ...
                [265 5 400 510], [670 5 325 510]};

            % panels
            hPan = gobj(1, numel(titles));
            for i=1:numel(titles)
                hPan(i) = uipanel('Parent',hParent, 'Title',titles{i}, ...
                    props{:}, 'Units','pixels', 'Position',pos{i});
            end
        end

        function createGUI_PanelHeader(this, hParent)
            %CREATEGUI_PANELHEADER  Create the header panel UI

            % dropdown menu properties and values
            props = {'ForegroundColor',0.5*[1 1 1], ...
                'HorizontalAlignment','left'};
            valsLR = bsxfun(@times, [1;3], 10.^(-5:1));
            valsLR = valsLR(:);
            valsAct = {'ReLU', 'Tanh', 'Sigmoid', 'Linear'};
            valsRegType = {'None', 'L1', 'L2'};
            valsReg = bsxfun(@times, [1;3], 10.^(-3:1));
            valsReg = [0; valsReg(:)];
            valsProb = {'Classification', 'Regression'};

            % run/step/reset buttons
            hBtnRun = uicontrol('Parent',hParent, 'Style','pushbutton', ...
                'String','Run', 'Units','pixels', 'Position',[10 45 60 20]);
            hBtnStep = uicontrol('Parent',hParent, 'Style','pushbutton', ...
                'String','Step', 'Units','pixels', 'Position',[10 25 60 20]);
            hBtnReset = uicontrol('Parent',hParent, 'Style','pushbutton', ...
                'String','Reset', 'Units','pixels', 'Position',[10 5 60 20]);

            % dropdown menus
            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Iterations', ...
                'Units','pixels', 'Position',[100 40 70 20]);
            hTxtIter = uicontrol('Parent',hParent, 'Style','text', ...
                'String','000', 'HorizontalAlignment','right', ...
                'FontSize',14, 'FontWeight','bold', ...
                'Units','pixels', 'Position',[100 15 70 25]);

            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Learning rate', ...
                'Units','pixels', 'Position',[210 40 120 20]);
            hPopLearnRate = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsLR, 'Value',8, ...
                'Units','pixels', 'Position',[210 15 120 20]);

            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Activation', 'TooltipString','Activation function', ...
                'Units','pixels', 'Position',[370 40 120 20]);
            hPopActFunc = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsAct, 'Value',2, ...
                'Units','pixels', 'Position',[370 15 120 20]);

            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Regularization', ...
                'Units','pixels', 'Position',[530 40 120 20]);
            hPopRegType = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsRegType, 'Value',1, ...
                'Units','pixels', 'Position',[530 15 120 20]);

            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Regularization rate', ...
                'Units','pixels', 'Position',[690 40 120 20]);
            hPopRegRate = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsReg, 'Value',1, ...
                'Units','pixels', 'Position',[690 15 120 20]);

            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Problem type', ...
                'Units','pixels', 'Position',[850 40 120 20]);
            hPopProblem = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsProb, 'Value',1, ...
                'Units','pixels', 'Position',[850 15 120 20]);

            % store handles
            this.handles.hBtnRun = hBtnRun;
            this.handles.hBtnStep = hBtnStep;
            this.handles.hBtnReset = hBtnReset;
            this.handles.hTxtIter = hTxtIter;
            this.handles.hPopLearnRate = hPopLearnRate;
            this.handles.hPopActFunc = hPopActFunc;
            this.handles.hPopRegType = hPopRegType;
            this.handles.hPopRegRate = hPopRegRate;
            this.handles.hPopProblem = hPopProblem;
        end

        function createGUI_PanelData(this, hParent)
            %CREATEGUI_PANELDATA  Create the data panel UI

            % properties
            props = {'ForegroundColor',0.5*[1 1 1], ...
                'HorizontalAlignment','left'};
            valsDatasets = {'Circle', 'XOR', 'Gaussian', 'Spiral'};

            % dataset dropdown menu
            uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String',{'Which dataset do','you want to use?'}, ...
                'Units','pixels', 'Position',[5 460 120 30]);
            hPopData = uicontrol('Parent',hParent, 'Style','popupmenu', ...
                'String',valsDatasets, 'Value',1, ...
                'Units','pixels', 'Position',[5 435 120 20]);

            % data sliders
            hTxtRatio = uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String',{'Ratio of training to','test data: 50%'}, ...
                'Units','pixels', 'Position',[5 385 120 30]);
            hSlidRatio = uicontrol('Parent',hParent, 'Style','slider', ...
                'Value',50, 'Min',10, 'Max',90, 'SliderStep',[5 10]./(90-10), ...
                'Units','pixels', 'Position',[5 360 120 20]);

            hTxtNoise = uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Noise: 10', ...
                'Units','pixels', 'Position',[5 315 120 20]);
            hSlidNoise = uicontrol('Parent',hParent, 'Style','slider', ...
                'Value',10, 'Min',0, 'Max',50, 'SliderStep',[5 10]./(50-0), ...
                'Units','pixels', 'Position',[5 295 120 20]);

            hTxtBatch = uicontrol('Parent',hParent, 'Style','text', props{:}, ...
                'String','Batch Size: 10', ...
                'Units','pixels', 'Position',[5 250 120 20]);
            hSlidBatch = uicontrol('Parent',hParent, 'Style','slider', ...
                'Value',10, 'Min',1, 'Max',30, 'SliderStep',[1 10]./(30-1), ...
                'Units','pixels', 'Position',[5 230 120 20]);

            % gen button
            hBtnGen = uicontrol('Parent',hParent, 'Style','pushbutton', ...
                'String','Regenerate', ...
                'Units','pixels', 'Position',[5 180 120 20]);

            % store handles
            this.handles.hPopData = hPopData;
            this.handles.hTxtRatio = hTxtRatio;
            this.handles.hSlidRatio = hSlidRatio;
            this.handles.hTxtNoise = hTxtNoise;
            this.handles.hSlidNoise = hSlidNoise;
            this.handles.hTxtBatch = hTxtBatch;
            this.handles.hSlidBatch = hSlidBatch;
            this.handles.hBtnGen = hBtnGen;
        end

        function createGUI_PanelInput(this, hParent)
            %CREATEGUI_PANELINPUT  Create the input panel UI

            % render thumnail image of each feature
            [X1,X2] = meshgrid(linspace(this.DOM(1),this.DOM(2),30));
            [Z,inputNames] = mapPoints([X1(:) X2(:)]);
            numInputs = numel(inputNames);
            Z = reshape(Z, [30 30 numInputs]);
            imgs = cell(1, numInputs);
            for i=1:numInputs
                imgs{i} = genThumbnail(Z(:,:,i), this.CMAP, false);
            end

            uicontrol('Parent',hParent, 'Style','text', ...
                'String',{'Which properties','do you want to','feed in?'}, ...
                'ForegroundColor',0.5*[1 1 1], 'HorizontalAlignment','left', ...
                'Units','pixels', 'Position',[5 445 100 45]);

            % input labels
            pos = cumsum([380 -45 -45 -45 -45 -45 -45]);
            for i=1:numInputs
                uicontrol('Parent',hParent, 'Style','text', ...
                    'String',inputNames{i}, ...
                    'HorizontalAlignment','right', 'FontWeight','bold', ...
                    'Units','pixels', 'Position',[5 pos(i) 50 20]);
            end

            % input toggle-buttons
            pos = cumsum([370 -45 -45 -45 -45 -45 -45]);
            hTBtnInput = gobj(1, numInputs);
            for i=1:numInputs
                hTBtnInput(i) = uicontrol('Parent',hParent, ...
                    'Style','togglebutton', 'CData',imgs{i}, 'Value',0, ...
                    'Units','pixels', 'Position',[60 pos(i) 40 40]);
            end
            set(hTBtnInput(1:2), 'Value',1);

            % store handles
            this.handles.hTBtnInput = hTBtnInput;
        end

        function createGUI_PanelLayers(this, hParent)
            %CREATEGUI_PANELLAYERS  Create the layers panel UI

            % layers components
            hBtnLayerAdd = uicontrol('Parent',hParent, ...
                'Style','pushbutton', 'String','+', ...
                'Units','pixels', 'Position',[130 465 20 20]);
            hBtnLayerDel = uicontrol('Parent',hParent, ...
                'Style','pushbutton', 'String','-', ...
                'Units','pixels', 'Position',[150 465 20 20]);
            hTxtLayerNum = uicontrol('Parent',hParent, 'Style','text', ...
                'String',pluralize(this.MAX_LAYERS, 'hidden layer'), ...
                'HorizontalAlignment','left', 'FontSize',12, ...
                'Units','pixels', 'Position',[175 465 120 20]);

            % neurons components per layer
            hBtnNeuronAdd = gobj(1, this.MAX_LAYERS);
            hBtnNeuronDel = gobj(1, this.MAX_LAYERS);
            hTxtNeuronNum = gobj(1, this.MAX_LAYERS);
            hBtnNeuron = gobj(this.MAX_NEURONS, this.MAX_LAYERS);
            img = genThumbnail(zeros(30), this.CMAP, false);
            pos = cumsum([0 75 75 75 75]);
            for k=1:this.MAX_LAYERS
                hBtnNeuronAdd(k) = uicontrol('Parent',hParent, ...
                    'Style','pushbutton', 'String','+', 'UserData',k, ...
                    'Units','pixels', 'Position',[30+pos(k) 435 20 20]);
                hBtnNeuronDel(k) = uicontrol('Parent',hParent, ...
                    'Style','pushbutton', 'String','-', 'UserData',k, ...
                    'Units','pixels', 'Position',[50+pos(k) 435 20 20]);
                hTxtNeuronNum(k) = uicontrol('Parent',hParent, ...
                    'Style','text', 'HorizontalAlignment','left', ...
                    'String',pluralize(this.MAX_NEURONS, 'neuron'), ...
                    'Units','pixels', 'Position',[25+pos(k) 415 60 20]);
                for n=1:this.MAX_NEURONS
                    hBtnNeuron(n,k) = uicontrol('Parent',hParent, ...
                        'Style','pushbutton', 'Enable','inactive', ...
                        'CData',img, 'Value',0, 'Units','pixels', ...
                        'Position',[30+pos(k) 370-(n-1)*45 40 40]);
                end
            end

            % store handles
            this.handles.hBtnLayerAdd = hBtnLayerAdd;
            this.handles.hBtnLayerDel = hBtnLayerDel;
            this.handles.hTxtLayerNum = hTxtLayerNum;
            this.handles.hBtnNeuronAdd = hBtnNeuronAdd;
            this.handles.hBtnNeuronDel = hBtnNeuronDel;
            this.handles.hTxtNeuronNum = hTxtNeuronNum;
            this.handles.hBtnNeuron = hBtnNeuron;
        end

        function createGUI_PanelOutput(this, hParent)
            %CREATEGUI_PANELOUTPUT  Create the output panel UI

            % properties
            props = {'XColor',0.5*[1 1 1], 'YColor',0.5*[1 1 1], ...
                'FontSize',8, 'LineWidth',0.5, 'Box','off'};
            [X1,X2] = meshgrid(linspace(this.DOM(1),this.DOM(2),250));
            clr = brighten(this.CMAP([1 end],:), -0.4);

            % axes loss (lines and legend)
            hC = uicontainer(hParent, ...
                'Units','pixels', 'Position',[10 390 300 100]);
            hAxLoss = axes('Parent',hC, props{:}, ...
                'ColorOrder',0.44*[0 0 0; 1 1 1], ...
                'Visible','off', 'XTick',[], 'YTick',[], ...
                'Units','pixels', 'Position',[25 5 250 70]);
            hLineLoss = line(NaN(1,2), NaN, 'LineWidth',1.5, ...
                'Parent',hAxLoss);
            hLgndLoss = legend(hLineLoss, ...
                'String',{'Test loss: 0.000', 'Training loss: 0.000'}, ...
                'FontSize',8, 'Interpreter','none', 'ButtonDownFcn','', ...
                'Location','NorthOutside', 'Orientation','Horizontal', ...
                'Units','pixels', 'Position',[25 80 250 20]);
            legend(hAxLoss, 'boxoff');

            % axes out (image, colorbar, and scatters)
            hC = uicontainer(hParent, ...
                'Units','pixels', 'Position',[10 60 300 325]);
            hAxOut = axes('Parent',hC, props{:}, ...
                'XLim',this.DOM, 'YLim',this.DOM, 'CLim',[-1 1], ...
                'XTick',this.DOM(1):1:this.DOM(2), ...
                'YTick',this.DOM(1):1:this.DOM(2), ...
                'TickDir','out', 'YDir','normal', ...
                'XAxisLocation','top', 'YAxisLocation','right', ...
                'Units','pixels', 'Position',[25 50 250 250]);
            hImgOut = image('Parent',hAxOut, ...
                'XData',X1(1,:), 'YData',X2(:,1), ...
                'CData',zeros(size(X1)), 'CDataMapping','scaled');
            hLineTrain = gobj(1,2);
            hLineTest = gobj(1,2);
            for i=1:2
                hLineTrain(i) = line('Parent',hAxOut, ...
                    'XData',NaN, 'YData',NaN, 'LineStyle','none', ...
                    'Marker','o', 'MarkerSize',6, 'LineWidth',0.5, ...
                    'MarkerFaceColor',clr(i,:), 'MarkerEdgeColor','w');
            end
            for i=1:2
                hLineTest(i) = line('Parent',hAxOut, 'Visible','off', ...
                    'XData',NaN, 'YData',NaN, 'LineStyle','none', ...
                    'Marker','o', 'MarkerSize',6, 'LineWidth',0.5, ...
                    'MarkerFaceColor',clr(i,:), 'MarkerEdgeColor','k');
            end
            if isHG1()
                orientH = {};
            else
                orientH = {'Orientation','horizontal'};
            end
            hCBar = colorbar('Peer',hAxOut, props{:}, ...
                orientH{:}, 'Location','SouthOutside', ...
                'XTick',-1:0.5:1, 'Color',0.5*[1 1 1], ...
                'Units','pixels', 'Position',[25 25 250 15]);
            set(hCBar, 'TickDir','out');

            % checkboxes
            hCBoxShowTest = uicontrol('Parent',hParent, ...
                'Style','checkbox', 'String','Show test data', ...
                'Value',0, 'Units','pixels', 'Position',[10 30 120 20]);
            hCBoxDiscretize = uicontrol('Parent',hParent, ...
                'Style','checkbox', 'String','Discretize output', ...
                'Value',0, 'Units','pixels', 'Position',[10 10 120 20]);

            % store handles
            this.handles.hLineLoss = hLineLoss;
            this.handles.hLgndLoss = hLgndLoss;
            this.handles.hImgOut = hImgOut;
            this.handles.hLineTrain = hLineTrain;
            this.handles.hLineTest = hLineTest;
            this.handles.hCBoxShowTest = hCBoxShowTest;
            this.handles.hCBoxDiscretize = hCBoxDiscretize;
        end

        function registerCallbacks(this)
            %REGISTERCALLBACKS  Setup event handlers for UI components

            set(this.handles.hBtnRun, 'Callback',@this.onRunPause);
            set(this.handles.hBtnStep, 'Callback',@this.onStep);
            set(this.handles.hBtnReset, 'Callback',@this.onReset);

            set([this.handles.hPopLearnRate, this.handles.hPopActFunc, ...
                this.handles.hPopRegType, this.handles.hPopRegRate, ...
                this.handles.hPopProblem, this.handles.hSlidBatch], ...
                'Callback',@this.onParamsChange);

            set([this.handles.hPopData,  this.handles.hSlidRatio, ...
                this.handles.hSlidNoise,  this.handles.hBtnGen], ...
                'Callback',@this.onDataChange);
            set(this.handles.hTBtnInput, 'Callback',@this.onInputChange);

            set([this.handles.hBtnLayerAdd, this.handles.hBtnLayerDel], ...
                'Callback',@this.onLayerAddRemove);
            set([this.handles.hBtnNeuronAdd, this.handles.hBtnNeuronDel], ...
                'Callback',@this.onNeuronAddRemove);

            set(this.handles.hCBoxShowTest, 'Callback',@this.onShowTest);
            set(this.handles.hCBoxDiscretize, 'Callback',@this.onDiscretize);
        end
    end

    %% UI Callbacks
    methods (Access = private)
        function onRunPause(this, ~, ~)
            %ONRUNPAUSE  Run/Pause buttons event handler

            % toggle run/pause
            this.isRunning = ~this.isRunning;
            if this.isRunning
                set(this.handles.hBtnRun, 'String','Pause');
            else
                set(this.handles.hBtnRun, 'String','Run');
            end

            % run in a loop
            while this.isRunning && ishghandle(this.handles.hFig)
                step(this);
                drawnow();
            end
        end

        function onStep(this, ~, ~)
            %ONRUNPAUSE  Step button event handler

            % state paused
            this.isRunning = false;
            set(this.handles.hBtnRun, 'String','Run');

            % run one pass
            step(this);
            drawnow();
        end

        function onReset(this, ~, ~)
            %ONRESET  Reset button event handler

            reset(this);
        end

        function onParamsChange(this, source, ~)
            %ONPARAMSCHANGE  Event handler for change in network options

            % update net params and UI as needed
            switch source
                case this.handles.hPopLearnRate
                    vals = cellstr(get(source, 'String'));
                    idx = get(source, 'Value');
                    this.net.LearningRate = str2double(vals{idx});
                case this.handles.hPopRegRate
                    vals = cellstr(get(source, 'String'));
                    idx = get(source, 'Value');
                    this.net.RegularizationRate = str2double(vals{idx});
                case this.handles.hSlidBatch
                    val = round(get(source, 'Value'));
                    set(this.handles.hTxtBatch, 'String', ...
                        sprintf('Batch Size: %2d',val));
                    this.net.BatchSize = val;
                otherwise
                    % recreate network
                    buildNet(this);
                    reset(this);
            end
        end

        function onDataChange(this, source, ~)
            %ONDATACHANGE  Event handler for change in data options

            % update UI labels as needed
            val = round(get(source, 'Value'));
            switch source
                case this.handles.hSlidRatio
                    set(this.handles.hTxtRatio, 'String', ...
                        {'Ratio of training to',...
                        sprintf('test data: %2d%%',val)});
                case this.handles.hSlidNoise
                    set(this.handles.hTxtNoise, 'String', ...
                        sprintf('Noise: %2d',val));
            end

            % generate new data
            genData(this);
            updatePlotScatter(this);

            % recreate network
            buildNet(this);
            reset(this);
        end

        function onInputChange(this, source, ~)
            %ONINPUTCHANGE  Input toggle buttons event handler

            % make sure we have at least one data feature selected
            inputsMask = getInputsMask(this);
            if ~any(inputsMask)
                set(source, 'Value',1);  % revert last change
                return;
            end

            % apply and store new data features
            this.data.inputs = mapPoints(this.data.points, inputsMask);

            % recreate network
            buildNet(this);
            reset(this);
        end

        function onLayerAddRemove(this, source, ~)
            %ONLAYERADDREMOVE  Layers add/remove buttons event handler

            % update hidden layers UI
            hiddenSizes = getHiddenSizes(this);
            switch get(source, 'String')
                case '+'
                    % add layer
                    if numel(hiddenSizes) >= this.MAX_LAYERS
                        return;
                    end
                    hiddenSizes(end+1) = 2;
                case '-'
                    % remove layer
                    if numel(hiddenSizes) <= 1
                        return;
                    end
                    hiddenSizes(end) = [];
            end
            updateHiddenLayers(this, hiddenSizes);

            % recreate network
            buildNet(this);
            reset(this);
        end

        function onNeuronAddRemove(this, source, ~)
            %ONNEURONADDREMOVE  Neurons add/remove buttons event handler

            % update hidden layers UI
            idx = get(source, 'UserData');  % hidden layer index
            hiddenSizes = getHiddenSizes(this);
            switch get(source, 'String')
                case '+'
                    % add neuron to specified layer
                    if hiddenSizes(idx) >= this.MAX_NEURONS
                        return;
                    end
                    hiddenSizes(idx) = hiddenSizes(idx) + 1;
                case '-'
                    % remove neuron from specified layer
                    if hiddenSizes(idx) <= 1
                        return;
                    end
                    hiddenSizes(idx) = hiddenSizes(idx) - 1;
            end
            updateHiddenLayers(this, hiddenSizes);

            % recreate network
            buildNet(this);
            reset(this);
        end

        function onShowTest(this, ~, ~)
            %ONSHOWTEST  Test data checkbox event handler

            % toggle test points visibility
            if get(this.handles.hCBoxShowTest, 'Value')
                val = 'on';
            else
                val = 'off';
            end
            set(this.handles.hLineTest, 'Visible',val);
        end

        function onDiscretize(this, ~, ~)
            %ONDISCRETIZE  Discretize checkbox event handler

            % if already running, no need to update plots as changes
            % will eventually get picked up on next iteration
            if ~this.isRunning
                updatePlotHeatmaps(this);
            end
        end
    end

end

%% Helper Functions

function b = isHG1()
    %ISHG1  Checks if running on HG1 or HG2 graphics
    b = verLessThan('matlab','8.4');
end

function H = gobj(varargin)
    %GOBJ  Create an array to store graphic handles

    try
        H = gobjects(varargin{:});
    catch
        H = zeros(varargin{:});
    end
end

function x = randU(a,b,varargin)
    %RANDU  Uniform random numbers

    try
        % Statistics Toolbox
        x = unifrnd(a, b, varargin{:});
    catch
        x = rand(varargin{:}) * (b-a) + a;
    end
end

function x = randMVN(mu, S, N)
    %RANDMVN  Random numbers from multivariate normal distribution

    try
        % Statistics Toolbox
        x = mvnrnd(mu, S, N);
    catch
        x = bsxfun(@plus, randn(N,numel(mu))*chol(S), mu); % cholcov
    end
end

function cmap = getColormap()
    %GETCOLORMAP  Return an orange-to-blue polarized colormap

    % interpolate between colors {'f59322', 'e8eaeb', '0877bd'}
    cmap = [245 147 34; 232 234 235; 8 119 189] ./ 255;
    cmap = interp1([-1 0 1], cmap, linspace(-1,1,256));

    % fake transparency
    a = 0.75;  % 160/255
    cmap = a*cmap + (1-a)*1.0;
end

function Z = input_X1(X1,~)
    Z = X1;
end

function Z = input_X2(~,X2)
    Z = X2;
end

function Z = input_X12(X1,~)
    Z = X1.^2;
end

function Z = input_X22(~,X2)
    Z = X2.^2;
end

function Z = input_X1X2(X1,X2)
    Z = X1.*X2;
end

function Z = input_sinX1(X1,~)
    Z = sin(X1);
end

function Z = input_sinX2(~,X2)
    Z = sin(X2);
end

function [inputs,names] = mapPoints(points, inputsIdx, scale)
    %MAPPOINTS  Map 2D points to inputs using specified features

    % all available feature functions and their labels
    funcs = {@input_X1, @input_X2, @input_X12, @input_X22, @input_X1X2, ...
        @input_sinX1, @input_sinX2};
    names = {'X1', 'X2', 'X1^2', 'X2^2', 'X1*X2', 'sin(X1)', 'sin(X2)'};

    % default values
    if nargin < 3, scale = false; end
    if nargin < 2, inputsIdx = true(size(funcs)); end

    % which functions to apply
    num = nnz(inputsIdx);
    funcs = funcs(inputsIdx);
    names = names(inputsIdx);

    % map 2D points to inputs
    inputs = zeros(size(points,1), num);
    for i=1:num
        inputs(:,i) = feval(funcs{i}, points(:,1), points(:,2));
    end

    % optional feature scaling
    if scale
        s = 6;
        scales = [1 1 1/s 1/s 1/s s s];
        inputs = bsxfun(@times, inputs, scales(inputsIdx));
    end
end

function [data,labels] = genDataCircle(N, noise)
    %GENDATACIRCLE  Generate 2D points for binary classification (two cocentric circles)

    radius = 5;
    data = cell(1,2);
    labels = cell(1,2);
    for i=1:2
        if i==1
            r = randU(0, radius*0.5, [N/2 1]);         % radii inside
        else
            r = randU(radius*0.7, radius, [N/2 1]);    % radii outside
        end
        t = rand(N/2,1) * 2*pi;                        % theta angles
        xy = bsxfun(@times, [cos(t) sin(t)], r);       % points
        data{i} = xy;
        nz = randU(-radius, radius, size(xy)) * noise; % add noise
        % labels: positive/negative points inside/outside the circle
        labels{i} = (hypot(xy(:,1)+nz(:,1), xy(:,2)+nz(:,2)) <= radius*0.5);
    end
    data = vertcat(data{:});            % 2D points in [-5,5]x[-5,5]
    labels = vertcat(labels{:})*2 - 1;  % labels: {0,1} -> {-1,1}
end

function [data,labels] = genDataXOR(N, noise)
    %GENDATAXOR  Generate 2D points for binary classification (Exclusive-OR shape)

    r = 5;
    data = rand(N,2)*2*r - r;               % 2D points in [-5,5]x[-5,5]
    data = data + sign(data)*0.3;           % padding away from origin
    nz = randU(-r, r, size(data)) * noise;  % add noise
    labels = (prod(data+nz,2) >= 0)*2 - 1;  % labels: {0,1} -> {-1,1}
end

function [data,labels] = genDataGaussian(N, noise)
    %GENDATAGAUSSIAN  Generate 2D points for binary classification (two gaussian blobs)

    % means and covariance
    mn = [2 2; -2 -2];
    sigma = eye(2) .* (noise * 7 + 0.5);  % [0.0,0.5] -> [0.5,4.0]
    % generate positive/negative samples from two gaussians
    data = [randMVN(mn(1,:), sigma, N/2); ...
            randMVN(mn(2,:), sigma, N/2)];
    labels = reshape(repmat([1 -1], N/2, 1), [], 1);
end

function [data,labels] = genDataSpiral(N, noise)
    %GENDATASPIRAL  Generate 2D points for binary classification (spiral shape)

    radius = 5;
    v = linspace(0, 1, N/2).';
    r = v * radius;       % radii
    t = v * 1.75 * 2*pi;  % angles
    data = cell(1,2);
    for k=1:2
        % phase shift
        if k == 1
            deltaT = 0;   % positive examples
        else
            deltaT = pi;  % negative examples
        end
        xy = bsxfun(@times, r, [sin(t + deltaT) cos(t + deltaT)]);  % points
        nz = randU(-1, 1, size(xy)) * noise;                        % add noise
        data{k} = xy + nz;
    end
    data = vertcat(data{:});  % 2D points in [-5,5]x[-5,5]
    labels = reshape(repmat([1 -1], N/2, 1), [], 1);
end

function IDX = splitData(labels, trainRatio)
    %SPLITDATA  Partition data

    % random partition
    % (without stratification, assumes two classes with equal proportions)
    N = numel(labels);
    %ind = randperm(N, round(trainRatio*N));
    ind = randperm(N);
    ind = ind(1:round(trainRatio*N));

    % train logical indices
    IDX = false(N,1);
    IDX(ind) = true;
end

function loss = mseLoss(Yhat, Y)
    %MSELOSS  Mean-squared error loss function
    loss = 0.5 * mean((Yhat - Y).^2);
end

function [stateAdd, stateDel] = getButtonState(num, mx)
    %GETBUTTONSTATE  Determine add/remove button on-off state based on current count

    % add button state
    if num >= mx
        stateAdd = 'off';
    else
        stateAdd = 'on';
    end

    % remove button state
    if num > 1
        stateDel = 'on';
    else
        stateDel = 'off';
    end
end

function img = genThumbnail(Z, cmap, discretize)
    %GENTHUMBNAIL  Render matrix into a truecolor heatmap image

    if nargin < 3, discretize = false; end
    if nargin < 2, cmap = getColormap(); end

    % discretize: [-1,1] range to {-1,+1} values
    if discretize
        Z = sign(Z);
    end

    % scale to 8-bit indexed image: [-1,1] -> [0,1] -> [0,255]
    try
        img = im2uint8(mat2gray(Z, [-1 1]));
    catch
        img = uint8(255 * (Z+1)/2);
    end

    % convert to truecolor according to colormap
    img = ind2rgb(img, cmap);
end

function str = pluralize(num, word)
    %PLURALIZE  Return formatted string(s) in singular/plural form according to count

    str = cell(size(num));
    for i=1:numel(num)
        if num(i) > 1
            suffix = 's';
        else
            suffix = '';
        end
        str{i} = sprintf('%d %s%c', num(i), word, suffix);
    end

    if isscalar(num)
        str = str{1};
    end
end
