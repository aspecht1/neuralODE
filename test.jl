using Catalyst, DiffEqFlux, DifferentialEquations, Plots #for gpu: CUDA
using Flux.Losses: mae
#using Flux.Optimise: update!

include("const.jl")
include("network.jl")

ps = [Tgas, user_crate, user_uv_ion, user_dust_to_gas_ratio, user_deff, fA, user_Tdust, user_Ghab, user_Av, user_H2self, user_COself]

u0 = zeros(length(states(small_network)))
u0[idx_H] = H
u0[idx_E] = E
u0[idx_Cj] = Ck
u0[idx_O] = O

tspan = (3e12, 3e16) # vielleicht von 0 starten?? und mehr steps??
t = 3*10 .^(range(12, stop=16, length=100)) # steps evenly log-sapced
t_end = tspan[2]

# ODE solution
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false))
prob = ODEProblem(small_network, u0, tspan, ps)
ode_data = Array(solve(prob, ode_solver, saveat=t, atol=1.f-6, rtol=1e-12))
yscale = maximum(ode_data, dims=2)[1:7] - minimum(ode_data, dims=2)[1:7]

# Neural Network to approximate the ODE
NN = Chain(x -> x,
          Dense(7, 5, gelu),
          Dense(5, 5, gelu),
          Dense(5, 5, gelu),
          Dense(5, 7)) #|> gpu

p, re = Flux.destructure(NN) # use this p as the initial condition!
re_NN = re(p) # need to restruture for backprop!   re_NN(u, p, t) = re(p)(u)

function dudt!(du, u, p, t) #scale the data
    du .= re_NN(u) .* yscale / t_end     #yscale und t_end auch auf gpu setzen???
end

prob = ODEProblem(dudt!, u0, tspan) # NeuralODE(NN, tspan, Tsit5(), saveat=t)
sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP())

function predict_n_ode()
  global re_NN = re(p) # WHY
  _prob = remake(prob, p=p, tspan=tspan) # WHY
  pred = Array(solve(_prob, ode_solver, u0=u0, p=p, saveat=t, atol=1.e-6, sensalg=sense))
  return pred
end

function loss_n_ode()
  pred = predict_n_ode()
  loss = mae(pred ./ yscale, ode_data ./ yscale)
  return loss
end

iter = 1
n_plot = 10
cb = function (;doplot=false) # callback function to observe training
  global iter
  #if iter % n_plot == 0
  pred = predict_n_ode()
  display(sum(abs2, ode_data .- pred)) # use mae function?
  # plot current prediction against data
  pl = scatter(t, ode_data[1, :], label="ODE solution")
  scatter!(pl, t, pred[1, :], label="prediction", xaxis=:log)
  display(plot(pl))
  #end
  #iter += 1
  return false
end

loss_n_ode()
cb() # Display the ODE with the initial parameter values.
opt = ADAMW(0.005, (0.9, 0.999), 1.f-6)
data = Iterators.repeated((), 100) #gpu?
@time Flux.train!(loss_n_ode, Flux.params(u0, p), data, opt, cb=cb)
