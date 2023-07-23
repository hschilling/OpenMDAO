import asyncio


async def process_packets():
    while True:
        packet = await receive_packet()
        # Process the first packet synchronously
        if packet == "Packet 0":
            print(f"Processing first packet: {packet}")
            await asyncio.sleep(3)  # Simulating processing delay
        else:
            asyncio.create_task(
                process_packet(packet))  # Create a task to process the packet asynchronously


async def receive_packet():
    # Code to receive packets from script A
    await asyncio.sleep(1)  # Receiving packets from Script A with one second delay b/w each packet
    packets = ["Packet 0", "Packet 1", "Packet 2", "Packet 3", "Packet 4"]
    for packet in packets:
        yield packet


async def process_packet(packet):
    print(f"Processing: {packet}")
    # do processing on each individual packet after the first packet
    print(f"Processed: {packet}")


asyncio.run(process_packets())