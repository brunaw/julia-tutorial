#-------------------------------------
# Code from presentation:
# Basic code ---
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

# Functions ---
function my_sum(x::Float64, y::Float64)
    x + y
end

# or
my_sum(x, y) = x + y

# let blocks
let x = 1; z = 2;
print(x + z)
end

# Parallelization ---
using Distributed # Package calling
nheads = @distributed (+) for i in 1:2000
    Int(rand(Bool))
end

# Plots ---
using Plots
using StatsPlots
gr()
x = 1:10; y = rand(10, 4)
p1 = plot(x, y) # Line Plot
p2 = scatter(x, y) # Scatter plot
p3 = boxplot(y, xlabel = "This one is labelled", title = "Subtitle")
p4 = histogram(y[:, 3]) # Histograms
all_plots = plot(p1, p2, p3, p4,
             layout = (2, 2), legend = false)

# Fancy plots ---
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

# The query package ---
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

# The MLJ package ---
using MLJ, StatsBase, DataFrames
using Plots
using XGBoost, MLJ

X, y = @load_crabs
X = DataFrame(X)

@load XGBoostClassifier
xgb  = XGBoostClassifier()
xgbm = machine(xgb, X, y)
r = range(xgb, :num_round, lower=10, upper=500)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=25,
                        measure=cross_entropy)

plot_curve = plot(curve.parameter_values,
    curve.measurements,
    xlab=curve.parameter_name,
    xscale=curve.parameter_scale,
    ylab = "CV estimate of accuracy")
#-------------------------------------
