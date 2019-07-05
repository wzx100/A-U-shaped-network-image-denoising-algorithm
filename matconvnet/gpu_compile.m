
vl_compilenn('enableGpu', true, ...
'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0', ...  %change it 
'cudaMethod', 'nvcc',...%windows系统下选这个模式
'enableCudnn','true',...
'cudnnroot','local/cuda');
%}
warning('off');