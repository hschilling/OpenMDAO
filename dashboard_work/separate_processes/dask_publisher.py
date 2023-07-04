from dask.distributed import Client, Pub

# Start a Dask distributed client
client = Client()

# Create a publisher
publisher = Pub('my-topic')

# Publish some data
data = [1, 2, 3, 4, 5]
for item in data:
    publisher.put(item)

# Close the client
client.close()
