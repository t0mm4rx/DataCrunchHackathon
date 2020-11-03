# DataCrunch Hackathon

## Goal

The goal is to predict the evolution of assets.

Contestants are ranked by precision: True Positive / (True Positive + False Positive)

## What I've done

I finished 8th out of 17, with a total precision of 78.26%.

The 6th has a score of 84.00%, so there is a huge gap I didn't manage to fill.

## What I have learned

- Before modeling, defining clear metrics is **necessary**. I struggled to understand which model was good and which was bad because I hadn't a clear goal.
- During modeling, **start with a small model, with small data**. A big problem I faced was that every training took like 20-30 minutes, so it was hard to create small models to test different strategies. Next time, I will create a few small models, see which one work best, then increase their size progressively. It should overfit first, then we try to make it more general.
- PCA makes your features uncorrelated.
- Gradient boosting can take raw data, not normalized or standardized.
- Neural network are sensitive to useless features, and keep them away from learning properly. Feature selection is important.
- Plotting training is important as it can say if we can increase the size, if we overfit, underfit...
- It's important to be sure that we fully use the GPU power with the tools we use. Catboost has an option to switch on manually. Don't forget to enable GPU in colab too.
