classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
    ignoreAverage = false
    opts = {}
  end

  properties (Transient)
      average = 0
      numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
        %%wzx add
        if obj.loss == 'l2'
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
            
        else obj.loss == 'l1'
            outputs{1} = vl_nnloss1(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
            
        end
         %%wzx add
%         outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
%         outputs{1} = vl_nnloss1(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
%       outputs{1} = vl_nnloss(inputs{1}, inputs{2}, inputs{3}, inputs{4}, [], 'loss', obj.loss, obj.opts{:}) ;%%% wzx add
%       last_loss = 0;
%       outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;%%% wzx add
%       last_loss = 1;
%       outputs{1} = vl_nnloss(inputs{3}, inputs{4}, [], 'loss', obj.loss, obj.opts{:},last_loss) ;%%% wzx add
        obj.accumulateAverage(inputs, outputs);
    end

    function accumulateAverage(obj, inputs, outputs)
        if obj.ignoreAverage, return; end;
        n = obj.numAveraged ;
        m = n + size(inputs{1}, 1) *  size(inputs{1}, 2) * size(inputs{1}, 4);
        obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
        if obj.loss == 'l2'
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
            
        else obj.loss=='l1'
            derInputs{1} = vl_nnloss1(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;
            
        end

%         derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;

        % derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, inputs{3}, inputs{4}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;;%%% wzx add
        derInputs{2} = [] ;
        derParams = {} ;
    end

    function reset(obj)
        obj.average = 0 ;
        obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
        outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
        % the receptive field depends on the dimension of the variables
        % which is not known until the network is run
        rfs(1,1).size = [NaN NaN] ;
        rfs(1,1).stride = [NaN NaN] ;
        rfs(1,1).offset = [NaN NaN] ;
        rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
        obj.load(varargin) ;
    end
  end
end
