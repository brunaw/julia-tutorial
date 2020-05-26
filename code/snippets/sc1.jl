# ----------------------------------------------------------------
# This script covers the basics & string stuff
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------

println("Starting with Julia today!")
# This is pi, you know
my_pi = 3.14159
typeof(my_pi)

# Basic operations -------------------------------
sum = 3 + 7
difference = 10 - 3
product = 20 * 5
quotient = 100 / 10
power = 10 ^ 2
modulus = 101 % 2

parse(Int64, "1")

# Strings -------------------------------
s1 = "This really is a string"

typeof('a') # Char

# String interpolation -------------------------------
name = "Jane"
num_fingers = 10
num_toes = 10
println("Hello, my name is $name.")
println("I have $num_fingers fingers and $num_toes toes.")
println("That is $(num_fingers + num_toes) fingers in all!!")

# String concatenation -------------------------------
s3 = "How many cats ";
s4 = "is too many cats?";
😺 = 10
string(s3, s4, " I don't think there is a limit.")
string("I don't know, but ", 😺, " is too few.")
# This also concatenates:
s3*s4

a = 3
b = 4
"$a + $b"
