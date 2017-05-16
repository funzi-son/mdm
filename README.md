# mdm

Mixed-dependency models for multi-resident activity recognition in smart homes [see mdm.pdf]

Requirements: Python 3.5+, Numpy.

How to run (Linux):

   run.sh <model_name> <data_name> <state_type> [alpha] [beta] [gamma]
   
     - model_name:
     
            + hmm     : group-dependency HMMs
            
            + xhmm    : grouped-dependency coupled HMMs
            
            + phmm    : parallel HMMs
            
            + chmm    : coupled HMMs
            
            + fhmm    : factorial HMMs
            
            + cd-fhmm : crossed-dependency factorial HMMs
            
            + md-hmm  : ensembles of HMMs
            
            + mdm     : mixed-dependency model
            
     - data_name:
     
            + casas: CASAS
            
            + arasa: ARAS House A
            
            + arasb: ARAS House B
            
     - state_type: dis , vec1, vec2, vec3
     
     - alpha, beta, gamma: must be set if model_name=mdm
     
  
Example:

./run.sh pmm casas dis

Results of  Model: hmm, Data:casas,State: dis

        R1    |   R2 |  All
          
Day  1      46.736   |   62.166   |   40.504

Day  2      71.193   |   74.365   |   57.487

Day  3      64.165   |   68.765   |   54.722

Day  4      80.000   |   84.967   |   75.817

Day  5      86.543   |   81.481   |   76.049

Day  6      84.692   |   77.969   |   70.243

Day  7      89.236   |   85.069   |   81.250

Day  8      83.752   |   79.524   |   77.279

Day  9      88.718   |   92.051   |   86.154

Day 10      82.395   |   78.066   |   73.593

Day 11      73.997   |   80.257   |   69.984

Day 12      77.273   |   83.741   |   72.028

Day 13      72.890   |   73.052   |   67.857

Day 14      73.034   |   75.120   |   65.971

Day 15      87.582   |   85.131   |   81.046

Day 16      73.770   |   85.792   |   69.945

Day 17      72.462   |   84.000   |   67.231

Day 18      69.371   |   90.350   |   66.993

Day 19      73.333   |   73.810   |   68.730

Day 20      72.760   |   61.290   |   57.885

Day 21      76.000   |   76.200   |   69.600

Day 22      90.806   |   73.065   |   69.194

Day 23      84.773   |   83.309   |   75.842

Day 24      90.166   |   85.325   |   80.635

Day 25      80.334   |   74.036   |   63.882

Day 26      65.589   |   66.051   |   57.044

==========================================

Average     77.368   |   78.267   |   69.114
