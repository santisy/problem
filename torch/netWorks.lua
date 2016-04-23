require "torch"
require "cutorch"
require "nn"
require "cunn"
require "optim"
require "./customLayers"


netWorks = {}

-- init the parameters
io.write("\027[1;31mhas normalizations\027[0m\n")
netWorks.initParameters = function(picDim, actionNum)
  netWorks.picDim = picDim
  netWorks.actionNum = actionNum
end



netWorks.buildNetQ = function ()
  local  picDim = netWorks.picDim
  local  actionNum = netWorks.actionNum
  local  qModel = nn.Sequential()
  do
    local parallelModel = nn.ParallelTable()

    local convModel = nn.Sequential()
    convModel:add(nn.CustomSpConv(3, 32, 5, 5))
    convModel:add(nn.SpatialBatchNormalization(4))
    convModel:add(nn.ReLU())
    -- convModel:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    convModel:add(nn.CustomSpConv(32, 32, 5, 5))
    convModel:add(nn.SpatialBatchNormalization(4))
    convModel:add(nn.ReLU())
    -- convModel:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    convModel:add(nn.CustomSpConv(32, 32, 5, 5))
    convModel:add(nn.SpatialBatchNormalization(4))
    convModel:add(nn.ReLU())
    -- convModel:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    local convResultsNum = (picDim - 5*3 + 1*3)^2*32

    convModel:add(nn.Reshape(convResultsNum))

    parallelModel:add(convModel)
    parallelModel:add(nn.Identity())



    qModel:add(parallelModel)
    qModel:add(nn.JoinTable(2)) 
    qModel:add(nn.BatchNormalization(2))
    qModel:add(nn.CustomLinear(convResultsNum+actionNum, 200))
    qModel:add(nn.BatchNormalization(2))
    qModel:add(nn.ReLU())
    qModel:add(nn.CustomLinear(200, 200))
    qModel:add(nn.BatchNormalization(2))
    qModel:add(nn.ReLU())

    -- the output layer
    local finalLayer = nn.Linear(200, 1)
    finalLayer.weight = torch.Tensor(1, 200):uniform(-3e-4, 3e-4)
    finalLayer.bias = torch.Tensor(1):uniform(-3e-4, 3e-4)

    qModel:add(finalLayer)
    qModel:add(nn.ReLU())
  end

  -- try cuda
  qModel:cuda()


  return qModel
end
