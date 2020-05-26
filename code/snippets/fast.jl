# ----------------------------------------------------------------
# Fast Julia
# Date: April, 2019
# ----------------------------------------------------------------
using BenchmarkTools
using Libdl
using Plots
using Statistics

a = rand(10^7) # 1D vector of random numbers, uniform on [0,1)
sum(a)

@time sum(a)

C_code = """
#include <stddef.h>
double c_sum(size_t n, double *X) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += X[i];
    }
    return s;
}
"""

const Clib = tempname()   # make a temporary file

open(`gcc -fPIC -O3 -msse3 -xc -shared -o $(Clib * "." * Libdl.dlext) -`, "w") do f
    print(f, C_code)
end

# define a Julia function that calls the C function:
c_sum(X::Array{Float64}) = ccall(("c_sum", Clib), Float64, (Csize_t, Ptr{Float64}), length(X), X)

c_sum(a)
c_sum(a) ≈ sum(a)
c_sum(a) - sum(a)

c_bench = @benchmark c_sum($a)

println("C: Fastest time was $(minimum(c_bench.times) / 1e6) msec")

d = Dict()  # a "dictionary", i.e. an associative array
d["C"] = minimum(c_bench.times) / 1e6  # in milliseconds
d


gr()
t = c_bench.times / 1e6 # times in milliseconds
m, σ = minimum(t), std(t)

histogram(t, bins = 500,
    xlim = (m - 0.01, m + σ), color = "orange",
    xlabel = "milliseconds", ylabel="count", label="")
