# ----------------------------------------------------------------
# Some data structures
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------

# Tuples -------------------------------
myfavoriteanimals = ("penguins", "cats", "sugargliders")

myfavoriteanimals[1]
# since tuples are immutable, we can't update it!
myfavoriteanimals[1] = "otters"

# NamedTuples -------------------------------
myfavoriteanimals = (bird = "penguins", mammal = "cats", marsupial = "sugargliders")
myfavoriteanimals.bird
myfavoriteanimals[2]

# Dictionaries -------------------------------
myphonebook = Dict("Jenny" => "867-5309", "Ghostbusters" => "555-2368")
myphonebook["Jenny"]
myphonebook["Kramer"] = "555-FILK"
myphonebook

# We can delete Kramer from our contact list - and simultaneously grab his number - by using pop!
pop!(myphonebook, "Kramer")

# Unlike tuples and arrays, dictionaries are not ordered. So, we can't index into them.
myphonebook[1]

# Arrays -------------------------------
# Unlike tuples, arrays are mutable. Unlike dictionaries, arrays contain ordered collections.
myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]
fibonacci = [1, 1, 2, 3, 5, 8, 13]
fibonacci[1]
mixture = [1, 1, 2, 3, "Ted", "Robyn"]
mixture[5]
myfriends[3] = "Baby Bop"

# We can add another number to our fibonnaci sequence
push!(fibonacci, 21)
# and then remove it
pop!(fibonacci)

# the following are arrays of arrays:
favorites = [["koobideh", "chocolate", "eggs"],["penguins", "cats", "sugargliders"]]
numbers = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
favorites[1][1]
numbers[1][3]

r = rand(4, 3)
r[1][1]
r2 = rand(4, 3, 2)
r2[2][1]

somemorenumbers = copy(fibonacci)
somemorenumbers[1] = 404
somemorenumbers
