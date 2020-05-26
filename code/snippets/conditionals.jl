# ----------------------------------------------------------------
# Conditionals
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------

N = 5
if (N % 3 == 0) && (N % 5 == 0)
    println("FizzBuzz")
elseif N % 3 == 0
    println("Fizz")
elseif N % 5 == 0
    println("Buzz")
else
    println(N)
end

x = 5
y = 3
if x > y
    x
else
    y
end

# as a ternary operator, the conditional looks like this:
(x > y) ? x : y

# short-circuit evaluation
false && (println("hi"); true)

# if a is true, Julia knows it can just return the value of b as the overall
# expression. This means that b doesn't necessarily need evaluate to true or false!
# b could even be an error:
(x > 0) && error("x cannot be greater than 0")

# or ||
true || println("hi")
x > 3 || println("hi")
(x > 0) && x > 6

# Write a conditional statement that prints a number if the number is even
# and the string "odd" if the number is odd
x = 4
if (x % 2 == 0)
    (x = x + 1; println("even, but not anymore, now $x"))
else
    print("odd")
end

# Rewrite the code using a ternary operator.

x = 4
(x % 2 == 0) && println("even")
x = x + 1
(x % 2 != 0) && println("odd") 
