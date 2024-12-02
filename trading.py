import torch


def naive_trading_strategy(model, test_data, sequence_length, criterion, item_ids, initial_balance=10000):
    model.eval()
    balance = initial_balance
    inventory = 0
    predicted_future_price = 0
    with torch.no_grad():
        for i in range(len(test_data) - sequence_length - 1):
            inputs = test_data[i:i + sequence_length]
            current_price = test_data[i, item_ids.index("566")].squeeze()[0]  # Current price
            print(f"current actual: {current_price}. predicted: {predicted_future_price}")

            # Model prediction
            outputs = model(inputs)
            predicted_future_price = outputs[0].item()  # Predicted price
            print(f"model predicted price: {outputs}")
            # Naive trading logic
            if predicted_future_price > current_price:
                # Buy condition: If we predict a rise in price and have enough balance
                if balance > current_price:
                    # purchase_price = current_price
                    inventory += 1
                    balance -= current_price
                    print(f"Bought 1 item at {current_price:.2f}, new balance: {balance:.2f}")
            elif inventory > 0 and predicted_future_price < current_price:
                # sell condition -- if our current price is higher than the predicted next price
                balance = balance + (current_price * inventory)
                print(f"Sold {inventory} items at {current_price:.2f}, new balance: {balance:.2f}")
                inventory = 0

    # Final balance and profit
    print(f"Final balance: {balance:.2f}, remaining inventory: {inventory}")
    print(f"Current worth of inventory: {current_price*inventory:.2f}")
    print(f"Balance + current worth: {balance+(current_price*inventory):.2f}")
