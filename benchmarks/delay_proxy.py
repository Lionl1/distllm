"""TCP proxy that delays every chunk by one-way latency (RTT/2) in each direction, preserving order."""
import asyncio, sys, time

listen, target, rtt_ms = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
one_way = rtt_ms / 2000

async def pipe(reader, writer):
    q = asyncio.Queue()
    async def rx():
        while data := await reader.read(1 << 20):
            await q.put((time.monotonic() + one_way, data))
        await q.put(None)
    async def tx():
        while (item := await q.get()) is not None:
            due, data = item
            if (d := due - time.monotonic()) > 0:
                await asyncio.sleep(d)
            writer.write(data); await writer.drain()
        writer.close()
    await asyncio.gather(rx(), tx(), return_exceptions=True)

async def handle(cr, cw):
    ur, uw = await asyncio.open_connection("127.0.0.1", target)
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw), return_exceptions=True)

async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", listen)
    async with srv: await srv.serve_forever()
asyncio.run(main())
