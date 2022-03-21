# Density Ratio Estimation for Selection Experiments

What is the right way to model high-throughput sequencing data arising from biological selection experiments? One common approach involves two steps: 1) pre-process the sequencing data to estimate some type of enrichment score for each sequence and 2) use these enrichment scores as the target for supervised learning. What if we, instead, recognize the enrichment score as a density ratio and attempt to estimate it directly? Does reframing this modeling problem as density ratio estimation have any benefits? Perhaps in terms of the robustness of the ranking of the sequences, or in applicability to different types of sequencing technologies?

See [the worklog](https://docs.google.com/document/d/10IupFgQpyiduU3cWUWYregTW2uuTfISdSs13cMQp4-8/edit#heading=h.6ljosr7dawi5) for this project for more information.
