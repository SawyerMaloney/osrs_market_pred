# Trying to make an OSRS market predictor
By Sawyer Maloney | sawyerdmaloney@gmail.com

### Steps
1. Get a lot of data in as small an increment as possible (5m increments thus far) -- changed this to 24h
2. Make predictions

### TODO
4. Serialize the model
5. Some function to test after training, preferrably showing loss as well as how much money we would make or loss

### Notes
    Changed the timestep to 24hrs instead of 5m because 5m seemed like a lot of random noise. 5m data is still stored in one of the files. Also, only grabbed like 5 items for the 24hrs timestep. Need to grab more, and then want to calculate correlation coefficient. Also required making changes to the get_item_data.py script so that we could use a different (and seemingly more efficient) method for getting information.

### Utilized Publications
https://www.sciencedirect.com/science/article/pii/S1746809424002933 -- pearson coefficient matrices
