using Pkg
# Pkg.add("DecisionTree")
using DecisionTree
using Statistics, Random

X = randn(1000, 2)
y = X * [2, 3] .+ 1

model = DecisionTreeRegressor(max_depth = 3)
f = fit!(model, X, y, max_features = 2)
f
print_tree(f)

# train depth-truncated classifier
model = DecisionTreeClassifier(max_depth=2)
fit!(model, features, labels)

d = Dict("a"=>1, "b"=>2, "c"=>3);

pop!(d, "a")
print(y)
