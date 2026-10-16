"""One-use delayed labels from the SAME ongoing online flight, not replay.

Future terminal feedback selects a supervision source ONLY. No outcome,
episode identity or future observation is ever supplied to the student.
"""
import numpy as np


class DelayedFailureLabels:
    def __init__(self, window_physics_steps):
        if int(window_physics_steps) != window_physics_steps or window_physics_steps <= 0:
            raise ValueError("Require a positive integer outcome window")
        self.window = int(window_physics_steps)
        self.pending = []
        self.previous_physics = -1
        self.seen = self.emitted = 0

    def push(self, batch, slots, episodes, physics, events):
        if physics <= self.previous_physics:
            raise ValueError("Online states must advance the physical clock")
        n = len(batch["target"])
        slots, episodes = np.asarray(slots), np.asarray(episodes)
        if (slots.shape != (n,) or episodes.shape != (n,)
                or slots.dtype.kind not in "iu" or episodes.dtype.kind not in "iu"
                or len(np.unique(slots)) != n or (slots < 0).any() or (episodes < 0).any()
                or any(len(x) != n or not np.isfinite(x).all() for x in batch.values())):
            raise ValueError("Malformed fresh state identities/data")
        checked = []
        for slot, episode, when in events:
            if (any(int(x) != x for x in (slot, episode, when))
                    or min(slot, episode) < 0 or not self.previous_physics < when <= physics):
                raise ValueError("Failure feedback is stale, future, or malformed")
            checked.append((slot, episode, when))
        for entry in self.pending:
            for slot, episode, when in checked:
                if entry["physics"] < when <= entry["physics"]+self.window:
                    entry["eligible"] |= ((entry["slots"] == slot) & (entry["episodes"] == episode))
        self.previous_physics = physics
        self.pending.append(dict(batch={k: np.array(v, copy=True) for k, v in batch.items()},
                                 slots=slots.copy(), episodes=episodes.copy(), physics=physics,
                                 eligible=np.zeros(n, dtype=bool)))
        self.seen += n
        ready = []
        while self.pending and self.pending[0]["physics"]+self.window <= physics:
            ready.append(self.pending.pop(0))
        result = {k: (np.concatenate([x["batch"][k] for x in ready], axis=0)
                       if ready else np.empty_like(v[:0])) for k, v in batch.items()}
        result["correction_eligible"] = (np.concatenate([x["eligible"] for x in ready])
                                         if ready else np.empty(0, dtype=bool))
        self.emitted += len(result["target"])
        return result

    @property
    def remaining(self):
        return sum(len(x["slots"]) for x in self.pending)
