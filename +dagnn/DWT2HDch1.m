classdef DWT2HDch1 < dagnn.ElementWise
    

  properties (Transient)
      padding = 0
      wavename = 'haart'
      opts = {}
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      outputs{1} = vl_nndwt2(inputs{1}, [], ...
          'wavename', obj.wavename, 'padding', obj.padding, obj.opts{:}) ;
      outputs{1}=outputs{1}(:,:,1,:) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} =  vl_nndwt2(inputs{1},  derOutputs{1}, ...
          'wavename', obj.wavename, 'padding', obj.padding, obj.opts{:}) ; 
      derInputs{1}=derInputs{1}(:,:,1,:);
      derParams = {0} ;
    end
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = {} ;
    end

    function rfs = getReceptiveFields(obj)
        rfs = [] ;
    end

    function obj = DWT2HD(varargin)
      obj.load(varargin) ;
    end
  end
end
