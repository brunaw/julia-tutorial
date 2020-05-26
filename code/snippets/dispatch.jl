# ----------------------------------------------------------------
# Multiple Dispatch
# Date: April, 2019
# ----------------------------------------------------------------
foo(x::String, y::String) = println("My inputs x and y are both strings!")
foo(x::Int, y::Int) = println("My inputs x and y are both integers!")
foo(3, 4)
foo("hi", "hello") # we didn't overwrite or replace the string option

methods(foo)
methods(+)

@which foo(3, 4) # To see which method is being dispatched

foo(x::Bool) = println("Now I'm bool")
foo(true)
@which foo(false)
#------
import Base: *, +, ^

*(α::Number,   g::Function) = x -> α * g(x)   # Scalar times function
*(f::Function, λ::Number)   = x -> f(λ * x)   # Scale the argument
*(f::Function, g::Function) = x -> f(g(x))    # Function composition  -- abuse of notation!  use \circ in Julia 0.6
^(f::Function, n::Integer) = n == 1 ? f : f*f^(n-1) # A naive exponentiation algorithm by recursive multiplication

using Plots

x = pi*(0:0.001:4)

plot(x, sin.(x),    c="black", label="Fun")
plot!(x, (12*sin).(x),    c="green", label="Num * Fun")
plot!(x, (sin*12).(x),    c="red", alpha=0.9, label="Fun * Num")
plot!(x, (5*sin*exp).(x), c="blue", alpha=0.2, label="Num * Fun * Fun")


plot([12*sin, sin*12, 5*sin*exp], 0:.01:4π, α=[1 .9 .2], c=[:green :red :blue])

x=(0:.01:2) * pi;

plot(x, (sin^2).(x), c="blue")     # Squaring just works, y=sin(sin(x)), Gauss would be pleased!
plot!(x, sin.(x).^2,  c="red")
