# Trying to make an OSRS market predictor

### TODO
1. Some function to test after training, preferrably showing loss as well as how much money we would make or loss
    - Train/test split needs to be done as well
    - So, MSE but also, say we start with 10,000 gp, what would we get after a naive trading strategy of buying when it predicts a raise in price over the next hour (since there is no shorting, this is the only way to trade)
2. Visualization of predictions for multiple timesteps would be nice
3. Larger cov/corrcoef analysis could also probably be good, so we could choose an item with a lot of information about

### Notes
    Changed the timestep to 24hrs instead of 5m because 5m seemed like a lot of random noise. 5m data is still stored in one of the files. Also, only grabbed like 5 items for the 24hrs timestep. Need to grab more, and then want to calculate correlation coefficient. Also required making changes to the get_item_data.py script so that we could use a different (and seemingly more efficient) method for getting information.

### Utilized Publications
https://www.sciencedirect.com/science/article/pii/S1746809424002933 -- pearson coefficient matrices
