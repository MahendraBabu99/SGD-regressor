Regression Models: SGD Regressor & Gradient Boosting Regressor

This repository demonstrates the implementation, comparison, and practical usage of two widely used regression models:

SGD Regressor (Stochastic Gradient Descent)

Gradient Boosting Regressor (GBR) with Prediction Intervals

The project focuses on tabular regression problems, model evaluation, and uncertainty estimation.

Project Overview

Regression models are often evaluated only on point predictions. However, in real-world applications, understanding uncertainty is equally important.

This repository covers:

Efficient learning on large datasets using SGD

Non-linear, high-accuracy modeling using Gradient Boosting

Construction and evaluation of prediction intervals using quantile regression

Models Included
1. SGD Regressor

SGD Regressor uses stochastic gradient descent to optimize a linear regression objective.

Key characteristics:

Scales well to large datasets

Supports online / incremental learning

Memory efficient

Requires feature scaling

Typical use cases:

Very large datasets

Streaming data

Approximate but fast solutions

2. Gradient Boosting Regressor

Gradient Boosting Regressor is an ensemble of decision trees trained sequentially to correct previous errors.

Key characteristics:

Handles non-linear relationships

Strong performance on tabular data

No feature scaling required

Not incremental

Typical use cases:

Medium-sized datasets

Complex feature interactions

High prediction accuracy

Prediction Intervals with Gradient Boosting

In addition to point predictions, this project demonstrates prediction interval estimation using quantile regression.

How it works:

Train separate models for lower and upper quantiles

Construct an interval 
[qlower​,qupper​]

Provides uncertainty-aware predictions
