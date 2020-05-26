# ----------------------------------------------------------------
# Loops
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------


myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]

let
  k = 0;
  n = 1000_000;
  @time for i in 1:n
         if (rand()^2 + rand()^2) < 1.0
           k += 1
         end
        end
        p = 4*k/n
end

let
  i = 0;
  while i < 10
    i += 1
    print(i)
 end
end

for i = 1:10
           global z
           z = z + i
end
