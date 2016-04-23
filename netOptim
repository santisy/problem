require "cutorch"
require "torch"
require "nn"
require "cunn"
require "netWorks"
require "optim"
require "gnuplot"

netOptim = {}


netOptim.train = function(inputs, action, qFormer)

  local criterion = nn.MSECriterion():cuda()
  -- adam parameters initiate
  local config = {}
  local state = {}
  config.learnigRate = 1e-3


  -- create closure to evaluate f(X) and df/dX
  local fNotes = torch.Tensor(200) -- note the loss

  local inputsTrain = inputs:clone()
  local actionTrain = action:clone()
  local qFormerTrain = qFormer:clone()

  for epoch = 1, 200 do

    local feval = function(x)
       -- collectgarbage()
       -- get new parameters
       if x ~= parameters then
          parameters:copy(x)
       end

       -- reset gradients
       gradParameters:zero()

       -- evaluate function for complete mini batch
       local outputs = qModel:forward{inputsTrain, actionTrain}
       local f = criterion:forward(outputs, qFormerTrain)
       fNotes[epoch] = f
       local df_do = criterion:backward(outputs, qFormerTrain)
       qModel:backward({inputsTrain, actionTrain}, df_do)

       -- Loss:
       f = f +  1e-2* torch.norm(parameters,2)^2/2

       -- Gradients:
       gradParameters:add( parameters:clone():mul(1e-2))

       -- return f and df/dX
       return f,gradParameters
    end
    optim.adam(feval, parameters, config, state)
  end
  -- print(timer:time().real)
  -- io.write("\n")
  gnuplot.plot(fNotes, '-')
end
