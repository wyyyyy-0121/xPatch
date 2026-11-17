"""Idle process used only to test the deadline freeze; no market data."""
import os,time
print(os.getpid(),flush=True)
while True:time.sleep(60)
