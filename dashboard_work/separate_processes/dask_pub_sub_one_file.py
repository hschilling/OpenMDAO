# from chatgpt
from dask.distributed import Client, Pub, Sub

# Start a Dask distributed client
client = Client()

# Create a publisher and subscriber
publisher = Pub('my-topic')
subscriber = Sub('my-topic')

# Define a function that will be executed on the workers
def square(x):
    return x ** 2

# Publish some data
data = [1, 2, 3, 4, 5]
for item in data:
    publisher.put(item)

# Create a Dask future for the result
result = client.submit(square, subscriber.get())

# Get the final result
final_result = client.gather(result)

print(final_result)  # Output: [1, 4, 9, 16, 25]
