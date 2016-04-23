require "torch"
require "nn"
require "cutorch"
require "cunn"

-- Initiate the custom weight
nn.CustomSpConv = function (nInputPlane, nOutputPlane, kW, kH)
  local conv_layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH)
  local bounds = torch.sqrt(nOutputPlane*kW*kH)
  conv_layer.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):uniform(-1/bounds, 1/bounds)
  conv_layer.bias = torch.Tensor(nOutputPlane):uniform(-1/bounds, 1/bounds)

  return conv_layer
end

nn.CustomLinear = function (inputDim, outputDim)
  local linear_layer = nn.Linear(inputDim, outputDim)
  local bounds = torch.sqrt(outputDim)
  linear_layer.weight = torch.Tensor(outputDim, inputDim):uniform(-1/bounds,1/bounds)
  linear_layer.bias = torch.Tensor(outputDim):uniform(-1/bounds, 1/bounds)
  return linear_layer
end
