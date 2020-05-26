# ----------------------------------------------------------------
# Functions
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------


function my_fc(name)
    println("Hi $name, it's great to see you!")
end
my_fc("Bruna")

# Alternatively, we could have declared either of these functions in a single line

my_fc_2(name) = println("Hi $name, it's great to see you!")
my_fc_2("Bruna")

squared(x) = x^2
squared(25)

# Finally, we could have declared these as "anonymous" functions
my_fc_3 = name -> println("Hi $name, it's great to see you!")
my_fc_3("Bruna")

squared_again = x -> x^2
squared_again(3)

# Duck-typing
A = rand(3, 3)
A
res = squared(A)
res[1, 2]


# f not work on a vector. Unlike A^2, which is well-defined,
# the meaning of v^2 for a vector, v, is not a well-defined algebraic operation.
v = rand(3)
squared(v)
squared(v[1])

# Mutating vs. non-mutating functions
v = [3, 5, 2]
println(sort(v))
# when we run sort!(v), the contents of v are sorted within the array v
sort!(v)
println(v)

# Some higher order functions
# map
sq = map(squared, [1, 2, 3])
sq[2]

map(x -> x^3, [1, 2, 3])[3]

# broadcast
# broadcast is a generalization of map, so it can do every thing map can do and more.
broadcast(squared, [1, 2, 3])

squared.([1, 2, 3])[3]

A = [i + 3*j for j in 0:2, i in 1:3]
# A * A
print(squared(A))

# All elements of A, squared
B = squared.(A)
print(B)

# This dot syntax for broadcasting allows us to write relatively complex compound
# elementwise expressions in a way that looks natural/closer to mathematical notation.
# For example, we can write
print(
A .+ 2 .* squared.(A) ./ A)

# Use map or broadcast to increment every element of matrix A by 1 and assign
# it to a variable A1.
A1 = broadcast(A -> A + 1, A)
print(A1)

A2 = A .+ 1
print(A2)
