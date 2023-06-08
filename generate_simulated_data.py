import random
import csv


def generate_simulated_data(num_periods, initial_price, volatility):
    price = initial_price
    data = []

    for _ in range(num_periods):
        price += random.uniform(-volatility, volatility)
        data.append(price)

    return data

currency_pair = 'EUR/USD'
num_periods = 1000
initial_price = 1.2
volatility = 0.01

price_data = generate_simulated_data(num_periods, initial_price, volatility)

# Save the simulated data to a CSV file
with open('simulated_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Currency Pair', 'Price'])
    for price in price_data:
        writer.writerow([currency_pair, price])
