# ag2/shop_mas.py
def call(self, receiver, kind, payload, expect):
    """Gửi một yêu cầu, chờ phản hồi đúng hợp đồng; lỗi thì ghi sự kiện và thử lại. Hết lượt -> None."""
    for attempt in range(self.max_retries + 1):
        msg = Message("coordinator", receiver, kind, payload)
        self._hop(msg)
        t0 = time.perf_counter()
        try:
            resp = self.agents[receiver].handle(msg)
            self.clock += 1000 * (time.perf_counter() - t0)
            self.check(resp, receiver, expect)
            self._hop(resp)
            return resp.payload
        except (TimeoutError, ContractError) as e:
            self._fault(msg, e, attempt)
    return None
