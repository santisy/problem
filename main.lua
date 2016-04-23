require "torch"
require "nn"
require "cutorch"
require "cunn"
require "customLayers"
require "netWorks"
require "netOptim"

netWorks.initParameters(60,3)
qModel = netWorks.buildNetQ()
inputs = torch.rand(16,3,60,60):cuda()
action = torch.rand(16, 3):cuda()
qFormer = torch.rand(16, 1):cuda()
parameters, gradParameters = qModel:getParameters()

print(qFormer)
netOptim.train(inputs, action, qFormer)
print(qFormer)   -- **qFormer changed, although it has been cloned**
