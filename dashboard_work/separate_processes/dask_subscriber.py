from dask.distributed import Client, Sub

# Start a Dask distributed client
client = Client()

# Create a subscriber
subscriber = Sub('my-topic')

# Define a function that will be executed on the workers
def square(x):
    return x ** 2

# Get the data from the subscriber
data = []
while subscriber.peek():
    item = subscriber.get()
    data.append(item)

# Apply the function on the data
result = [square(x) for x in data]

print(result)  # Output: [1, 4, 9, 16, 25]

# Close the client
client.close()
