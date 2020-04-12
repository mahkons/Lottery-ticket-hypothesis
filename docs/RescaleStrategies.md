# Rescale strategies

Globale rescale -- same scaling factor for every layer  
Local rescale -- same scaling factor for every neuron  


I -- initial weights, N -- weights after pruning  
L1 -- scaling_factor = I.abs().sum() / N.abs().sum()  
L2 -- scaling_factor = (I ** 2).sum().sqrt() / (N ** 2).sum().sqrt()  

L1 makes sense because ...  
L2 makes sense because ...  

[L1-L2 Local](https://drive.google.com/open?id=1kRLOlSzuF06sNAOaU8PX54PaZIDJ-upl)  
[L1-L2 Global](https://drive.google.com/open?id=1W04zdKi0vKIkFqANoMdQK8RrP125WTBw)  

L2 better than L1  

[L2Local - L2Global](https://drive.google.com/open?id=14woQ3Pnn8_rePpqj-tTlZXy6E8Q8S7K8)  
[L2Global - NoRescale](https://drive.google.com/open?id=146EBWx1Jr2LZVd5WfZw7MfbK-Ci_uo_F)  

L2Local, L2Global, NoRescale give almost same results  
Rescale may be better with extreme pruning rates (BigNet experiments)  
Further experiments TODO   

