# Julia Tutorial

## [Presentation](http://brunaw.com/julia-tutorial/slides/julia.html)
Roadmap
  - What is Julia
  - `R`, `python` and `Julia` comparison 
  - Introduction to Julia
  - Julia for DS
  - Julia for ML
  
  
## `R`, `python` and `Julia` comparison  

Code at: 
  [`R`](https://github.com/brunaw/julia-tutorial/blob/master/code/EM/EM.R), 
[`python`](https://github.com/brunaw/julia-tutorial/blob/master/code/EM/em.py), 
[`Julia`](https://github.com/brunaw/julia-tutorial/blob/master/code/EM/EM.jl)
```  
# Runs: Average 9 for each algorithm
# Results: 
#      - Julia apprx 243 times faster than R;
#      - Julia apprx 75 times faster than Python;
#  
#--- Time ------- R ------- Python ------ Julia  ---#
#  minimum: | 138.319 s  | 41.121  s |  535.408 ms  #
#  median:  | 152.872 s  | 46.800  s |  599.821 ms  #
#  mean:    | 152.534 s  | 46.705  s |  627.756 ms  #
#  maximum :| 173.347 s  | 53.734  s |  802.406 ms  #
#---------------------------------------------------#
```

### Update: 31 May 2020

Now we also have an optimized version of the python code at [this link](https://github.com/brunaw/julia-tutorial/blob/master/code/EM/em_numba.py) using numba's just-in-time compilation, adapted by [@luk-f-a](https://github.com/luk-f-a). Please see
[the pull request description](https://github.com/brunaw/julia-tutorial/pull/2) for details on the changes and the perfomance of the adapted code. 

## Useful Links

- [Learning resources](https://julialang.org/learning/)
- [Code snippets](https://github.com/brunaw/julia-tutorial/tree/master/code/snippets)

- [List of packages](https://juliaobserver.com/packages)
- [Julia course in Coursera](https://www.coursera.org/learn/julia-programming)

- [Tutorials](https://github.com/JuliaComputing/JuliaBoxTutorials)


Slides theme based on: https://github.com/rstudio-education/arm-workshop-rsc2019/tree/36f9665c8fb0b5cc0213a6c0786b6a1f8d0290fb/static/slides