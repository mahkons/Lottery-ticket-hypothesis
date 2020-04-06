# Compare Pruners

GlobalPruner -- prunes weights with the smallest magnitude  
LayerwisePruner -- prunes weights with the smallest magnitude in each layer. Does not prune biases  
ERPruner -- like layerwise, but prunes big layers more, than small ones. Does not prune biases

Plots averaged over four experiments  

![100%](/docs/Compare100.png)  
![51%](/docs/Compare51.png)  
![26%](/docs/Compare26.png)  
![13%](/docs/Compare13.png)  
![6.8%](/docs/Compare6.8.png)  
![4.4%](/docs/Compare4.4.png)  


Differences become noticeable in later iterations  
ERpruner performs better than LayerwisePruner  
Unexpectedly GlobalPruner performs better than both other pruners, even though it prunes biases, unlike other pruners  
