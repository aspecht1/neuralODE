using Catalyst, DiffEqFlux, DifferentialEquations, ForwardDiff, LinearAlgebra, Plots
using Flux.Losses: mae
using Flux.Optimise: update!

include("const.jl")
include("network.jl")

ps = [Tgas, user_crate, user_uv_ion, user_dust_to_gas_ratio, user_deff, fA, user_Tdust, user_Ghab, user_Av, user_H2self, user_COself]

u0 = zeros(length(states(small_network)))
u0[idx_H] = H
u0[idx_E] = E
u0[idx_Cj] = Ck
u0[idx_O] = O

tspan = (3e12, 3e16)
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
          Dense(5, 7))

p, re = Flux.destructure(NN)
re_NN = re(p)

function dudt!(du, u, p, t) #scale the data
    du .= re_NN(u) .* yscale / t_end
end

prob = ODEProblem(dudt!, u0, tspan)
sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP())

function predict_n_ode(p)
  global re_NN = re(p)
  _prob = remake(prob, p=p, tspan=tspan)
  pred = Array(solve(_prob, ode_solver, u0=u0, p=p, saveat=t, atol=1.e-6, sensalg=sense))
  return pred
end

function loss_n_ode(p)
  pred = predict_n_ode(p)
  loss = mae(pred ./ yscale, ode_data ./ yscale)
  return loss
end

cb = function (p;doplot=false) # callback function to observe training
  pred = predict_n_ode(p)
  display(sum(abs2, ode_data .- pred))
  # plot current prediction against data
  pl = scatter(t, ode_data[1, :], label="ODE solution")
  scatter!(pl, t, pred[1, :], label="prediction", xaxis=:log)
  display(plot(pl))
  return false
end

loss_n_ode(p)
cb(p) # Display the ODE with the initial parameter values.
opt = ADAMW(0.005, (0.9, 0.999), 1.f-6)
grad_max = 1.e2
for i = 1:1000
    global p
    loss = loss_n_ode(p)
    grad = ForwardDiff.gradient(x -> loss_n_ode(x), p)
    grad_norm = norm(grad, 2)
    grad = grad ./ grad_norm .* grad_max
    update!(opt, p, grad)
    cb(p)
end
