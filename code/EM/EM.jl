using Colors
using Plots
using DataFrames
using DelimitedFiles
using LinearAlgebra
using IterativeSolvers
using Distributions

# Data -------
samples = readdlm("samples.txt")
X = [[samples[setdiff(1:end, 1), 2]];
     [samples[setdiff(1:end, 1), 3]]]
pi = [0.34518140, 0.6066179 ,  0.04820071]    # The priors for the mixture proportions
mu = [[-0.7133619    0.90635089];
      [0.7662367   0.8260541];
      [ -1.3236828 -1.7524445]]               # The start values for \mu_c
sigma = cat(
       [1.004904 1.899802; 1.899802 4.183546],
       [1.9686781 0.7841534; 0.7841534 1.8331994],
       [0.1931634 -0.1164864; -0.1164864 1.9839597],
        dims = 3)   # Covariance matrix between the clusters
# Plotting -------
plot(X[1], X[2], seriestype=:scatter, label = "", title = "EM data")
# Main functions -------
#---------------------------------------------------------------
function E_step(X, pi, mu, sigma, C = 3)
    N = size(X[1])[1]
    d = size(mu)[2]
    gamma = zeros(Float64, N, C)
    x = zeros(Float64, N, d)

    for c in 1:C
        # Normalization constant of a biv. Normal distribution
        cst = ((2 * 3.141592)^d * det(sigma[:, :, c]))^0.5
        # Subtracts mu's of the observations
        for i in 1:d
            x[:, i] = [X[i] .- mu[c, i]][1]
        end

    # Calculating the core of the biv. Normal distribution
        xt = LinearAlgebra.transpose(x)[:, :]
        xinvSxt = x * inv(sigma[:, :, c]) .* x
        xinvSx = (sum(sum(xinvSxt, dims = 2), dims = N)) .* -1/2
        dens = exp.(xinvSx)/cst
        gamma[:, c] = pi[c] * dens
    end
    gamma_div = sum(gamma, dims = 2)
    final_gamma = gamma ./ gamma_div
    return final_gamma
end
#----------------------------------------------------------------
function M_step(X, gamma)
    N = size(X[1])[1] # number of objects
    C = size(gamma)[2] # number of clusters
    d = size(X)[1] # dimension of each object
    mu = zeros(Float64, C, d)
    sigma = zeros(Float64, d, d, C)
    gamma_sum = sum(gamma, dims=1)
    pi = gamma_sum/N
    x = zeros(Float64, N, d)
    for c in 1:C
        for m in 1:d
        mu[c, m] = sum(gamma[:, c] .* X[m, :][1])/gamma_sum[c]
        end
        for i in 1:d
            x[:, i] = [X[i] .- mu[c, i]][1]
        end
        sigma[:, :, c] = (transpose(x) * Diagonal(gamma[:, c]) * x)/gamma_sum[c]
    end
return pi, mu, sigma
end
#---------------------------------------------------------------------
function compute_vlb(X, _pi, mu, sigma, gamma, C = 3)
    N = size(X[1])[1]
    d = size(mu)[2]
    loss = zeros(Float64, 1)
    for c in 1:C
        for n in 1:N
            mv_norm = MvNormal(mu[c, :], Symmetric(sigma[:, :, c]))
            logd = logpdf(mv_norm, [X[1][n], X[2][n]])
            gamma_c = gamma[n, c]
            loss_here   = gamma_c * ((log(_pi[c]) + logd) - log(gamma_c))
            loss[1] = loss[1] + loss_here[1]
        end
    end
return loss
end
# Testing it
gamma = E_step(X, pi, mu, sigma)
pi, mu, sigma = M_step(X, gamma)
vlb = compute_vlb(X, pi, mu, sigma, gamma)
print(vlb)
#----------------------------------------------------------------
function train_EM(X, C, rtol = 1e-3, max_iter = 100, restarts = 25)
    N = size(X[1])[1] # number of objects
    d = size(mu)[2] # dimension of each object
    best_loss = -Inf
    best_pi = nothing
    best_mu = nothing
    best_sigma = nothing
    for _ in 1:restarts
            pi = rand(C)
            pi = pi/sum(pi)
            mu = reshape(randn(C * d), (C, d))
            sigma = zeros(Float64, (d, d, C))
            for c in 1:C
                sigma[:, :, c] = Diagonal(ones(d))
            end
            lloss = -Inf
            not_saturated = true
            iteration = 0
            while (not_saturated & (iteration < max_iter))
                former_loss = copy(lloss)
                gamma = E_step(X, pi, mu, sigma)
                pi, mu, sigma = M_step(X, gamma)
                lloss = compute_vlb(X, pi, mu, sigma, gamma)
                if(former_loss[1] > lloss[1])
                    print("bug")
                end
                not_saturated = abs((lloss[1]/former_loss[1]) - 1) > rtol
                iteration = iteration + 1
            end
            if(lloss[1] > best_loss[1])
                best_loss = copy(lloss)
                best_pi = pi
                best_mu = mu
                best_sigma = sigma
            end
    end
    return best_loss, best_pi, best_mu, best_sigma
end
# Training the model -------
best_loss, best_pi, best_mu, best_sigma = train_EM(X, 3)
gamma = E_step(X, best_pi, best_mu, best_sigma)
function indmaxC(x::AbstractArray)
    pos = zeros(Float64, size(x, 1))
    for c in 1:size(x, 1)
    pos[c] = findmax(x[c,:])[2]
    end
    return pos
end
labels = indmaxC(gamma)

plot(X[1], X[2], seriestype=:scatter, groups = labels,
label = "", title = "EM data, now with labels")

using BenchmarkTools
@benchmark train_EM(X, 3)

# BenchmarkTools.Trial:
#  memory estimate:  429.08 MiB
#  allocs estimate:  10251556
#  --------------
#  minimum time:     535.408 ms (14.89% GC)
#  median time:      599.821 ms (16.52% GC)
#  mean time:        627.756 ms (15.83% GC)
#  maximum time:     802.406 ms (16.11% GC)
#  --------------
#  samples:          8
#  evals/sample:     1



sum_for = 0;
value = 3;
for j in 1:5
if (value < 5)
sum_for += 1
elseif (value == 5)
sum_for += 2
else
sum_for = sum_for/2
    end
    global sum_for
    global value = value^sum_for
end


function my_sum(x, y)
    x + y
end

    my_sum(x, y) = x + y
# maths


using Distributed # Package calling
nheads = @distributed (+) for i in 1:2000
    Int(rand(Bool))
end


let x = 1; z = 2;
print(x + z)
end

using Query, DataFrames, RDatasets
cars = dataset("datasets", "mtcars")

df = cars |>
  @filter(_.MPG > 15) |>
  @groupby(_.Cyl) |>
   @map({Key=key(_), Count=length(_)}) |>
  DataFrame

 df
# 3×2 DataFrame
# │ Row │ Key   │ Count │
# │     │ Int64 │ Int64 │
# ├─────┼───────┼───────┤
# │ 1   │ 6     │ 7     │
# │ 2   │ 4     │ 11    │
# │ 3   │ 8     │ 8     │

using Plots
default(legend = false)
x = y = range(-5, 5, length = 40)
zs = zeros(0, 40)
n = 100

my_gif = @animate for i in range(0, stop = 2π, length = n)
    f(x, y) = sin(x + 10sin(i)) + cos(y)

    # create a plot with 3 subplots and a custom layout
    l = @layout [a{0.7w} b; c{0.2h}]
    p = plot(x, y, f, st = [:surface, :contourf], layout = l)

    # induce a slight oscillating camera angle sweep, in degrees (azimuth, altitude)
    plot!(p[1], camera = (10 * (1 + cos(i)), 40))

    # add a tracking line
    fixed_x = zeros(40)
    z = map(f, fixed_x, y)
    plot!(p[1], fixed_x, y, z, line = (:black, 5, 0.2))
    vline!(p[2], [0], line = (:black, 5))

    # add to and show the tracked values over time
    global zs = vcat(zs, z')
    plot!(p[3], zs, alpha = 0.2, palette = cgrad(:blues).colors)
end

gif(my_gif, "anim.gif", fps = 15)


Δ = 1
Σ = 3
α = 20

# Basis

# ML

using MLJ
using MLJModels
using DecisionTree
import MLJ.MLJModels.DecisionTree_.DecisionTreeClassifier
import XGBoost

tree_model = @load XGBoostClassifier verbosity=1
y, X = unpack(iris, ==(:Species), colname -> true);
train, test = partition(eachindex(y), 0.7, shuffle=true)
fit = machine(tree_model, X[train,:], y[train])
evaluate(fit, X[train,:], y[train])
MLJ.predict
XGBoost.predict

rfResultXY = machine(RandomForestClassifier, X[train,:], y[train])
fit!(rfResultXY)
ŷ_rfResultXY = MLJ.predict(rfResultXY, Xy)
ŷMode_rfResultXY = [mode(ŷ_rfResultXY[i]) for i in 1:length(ŷ_rfResultXY)]
return ((ŷMode_rfResultXY = ŷMode_rfResultXY, rfResultXY = rfResultXY))


@everywhere @load RandomForestClassifier  pkg = DecisionTree

yhat = XGBoost.predict(fit, X[test,:]);

DecisionTreeClassifier(
    max_depth = -1,
    min_samples_leaf = 1,
    min_samples_split = 2,
    min_purity_increase = 0.0,
    n_subfeatures = 0,
    post_prune = false,
    merge_purity_threshold = 1.0,
    pdf_smoothing = 0.0,
    display_depth = 5)

using MLJ
iris = load_iris()
@load DecisionTreeClassifier
tree_model = XGBoostClassifier(max_depth=2);
tree = machine(tree_model, X, y);
tree
fit!(tree, rows=train)

iris = RDatasets.dataset("datasets", "iris");

tree = machine(tree_model, X, y)
fit!(tree, rows=train);


import MLJModels ✔
import DecisionTree ✔
import MLJ.MLJModels.DecisionTree_.DecisionTreeClassifier ✔
DecisionTreeClassifier(
    max_depth = -1,
    min_samples_leaf = 1,
    min_samples_split = 2,
    min_purity_increase = 0.0,
    n_subfeatures = 0,
    post_prune = false,
    merge_purity_threshold = 1.0,
    pdf_smoothing = 0.0,
    display_depth = 5)


using MLJ
rang

using MLJModels
using DecisionTree
tree = @load DecisionTreeClassifier
tree_classifier= @pipeline Tree_Model(model = tree,
                                      yhat -> mode.(yhat))

ranges = range(tree,
               :(model.min_purity_increase);
               lower=0.01,
               upper=1.0,
               scale = :log)
self_tuning_tree_classifier = TunedModel(; model=tree_classifier,
                                        tuning=Grid(),
                                        resampling=CV(nfolds=5, shuffle=true),
                                        measure=accuracy,
                                        ranges=ranges)
self_tuning_tree = machine(self_tuning_tree_classifier, X, y)

n = 100000
X = DataFrame(A = rand(n), B = categorical(rand(1:10, n)))
y = categorical(rand(0:1, n))

@load XGBoostClassifier
xgb_machine = machine(XGBoostClassifier(), X, y)
fit_model = MLJ.fit!(xgb_machine; verbosity = 0)
fit_model
predictions = MLJ.predict(xgb_machine, X)
predictions
sum((predictions .> 0.5) .!= y)

y

cv=CV(nfolds=3)
evaluate(xgb_machine, X, y, resampling=cv, measure=l2, verbosity=0)



model = @load RidgeRegressor pkg=MultivariateStats
mach = machine(model, X, y)
MLJ.fit!(mach, X, y)

evaluate(model, X, y, resampling=cv, measure=l2, verbosity=0)
MLJ.predict(mach, X)
#MLJ.save("xgboost.jlso", xgb_machine)



using Pkg; Pkg.activate(".")
using MLJ, StatsBase, Random, CategoricalArrays, PrettyPrinting, DataFrames
X, y = @load_crabs
X = DataFrame(X)

using XGBoost, MLJ
@load XGBoostClassifier
xgb  = XGBoostClassifier()
xgbm = machine(xgb, X, y)
r = range(xgb, :num_round, lower=10, upper=500)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=25,
                        measure=cross_entropy)

using Plots
                        plot(curve.parameter_values,
                             curve.measurements,
                             xlab=curve.parameter_name,
                             xscale=curve.parameter_scale,
                             ylab = "CV estimate of RMS error")
