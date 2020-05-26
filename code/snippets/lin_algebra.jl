# ----------------------------------------------------------------
# Linear algebra & factorization
# Date: April, 2019
# ----------------------------------------------------------------

A = rand(1:4,3,3)
length(A)

# vector of ones
x = fill(1.0, (3,))

# multiplication
b = A * x
A'
# or
transpose(A)

A'A

# Solving linear systems
# least squares solution
A\b

Atall = rand(3, 2)

Atall\b

v = rand(3)
rankdef = hcat(v, v)
print(rankdef\b)

# minimum norm solution
bshort = rand(2)
Ashort = rand(2, 3)
Ashort\bshort

#--------------------------------------------------------
using LinearAlgebra
A = rand(3, 3)
x = fill(1, (3,))
b = A * x

# LU factorization
Alu = lu(A)
typeof(Alu)
Alu.P
Alu.L
Alu.U

# We can solve the linear system using either the original matrix
# or the factorization object.
print(
    A\b,
    Alu\b)

det(A) ≈ det(Alu)

# QR factorization
Aqr = qr(A)
print( Aqr.Q )

# Eigendecomposition
Asym = A + A'
AsymEig = eigen(Asym)
print(AsymEig.values, AsymEig.vectors, inv(AsymEig)*Asym)

# Special matrix structures
n = 1000
A = randn(n,n);
# Julia can often infer special matrix structure
Asym = A + A'
issymmetric(Asym)

# but sometimes floating point error might get in the way.
Asym_noisy = copy(Asym)
Asym_noisy[1,2] += 5eps()
issymmetric(Asym_noisy)


Asym_explicit = Symmetric(Asym_noisy);
# Let's compare how long it takes Julia to compute the eigenvalues of Asym, Asym_noisy, and Asym_explicit
@time eigvals(Asym);
@time eigvals(Asym_noisy);
@time eigvals(Asym_explicit);

# A big problem
n = 1_000_000;
A = SymTridiagonal(randn(n), randn(n-1));
@time eigmax(A)

# Rational numbers
1//2

# Example: Rational linear system of equations
Arational = Matrix{Rational{BigInt}}(rand(1:10, 3, 3))/10

x = fill(1, 3)
b = Arational*x
Arational\b
lu(Arational)
