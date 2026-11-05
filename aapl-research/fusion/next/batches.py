"""Bounded CPU preparation ahead of CUDA; this queue never owns training RNG."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class BatchAhead:
    def __init__(self,prepare,items,depth=2):
        if depth<1:raise ValueError('Prefetch depth must be positive')
        self.prepare=prepare;self.items=iter(items);self.depth=depth;self.pool=None

    def __enter__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='causal-batches');self.pending=deque()
        for _ in range(self.depth):self._submit()
        return self

    def _submit(self):
        try:item=next(self.items)
        except StopIteration:return
        self.pending.append(self.pool.submit(self.prepare,item))

    def __iter__(self):return self

    def __next__(self):
        if not self.pending:raise StopIteration
        result=self.pending.popleft().result();self._submit();return result

    def __exit__(self,*error):
        self.pool.shutdown(wait=True,cancel_futures=True)


def pin(batch,enabled):
    return {name:value.pin_memory() for name,value in batch.items()} if enabled else batch


def transfer(batch,device):
    return {name:value.to(device,non_blocking=device=='cuda') for name,value in batch.items()}
