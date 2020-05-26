# ----------------------------------------------------------------
# Plotting
# Author: Bruna Wundervald
# Date: April, 2019
# ----------------------------------------------------------------

using Colors
using Plots

globaltemperatures = [14.4, 14.5, 14.8, 15.2, 15.5, 15.8]
numpirates = [45000, 20000, 15000, 5000, 400, 17];

# A simple scatterplot
gr()
plot(numpirates, globaltemperatures, label = "line")
scatter!(numpirates, globaltemperatures, label = "points")
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")
# xflip!()

#-----------------------------------
unicodeplots()

plot(numpirates, globaltemperatures, label="line")
scatter!(numpirates, globaltemperatures, label="points")
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")
