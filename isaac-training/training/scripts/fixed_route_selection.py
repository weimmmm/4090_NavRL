"""Select existing fixed routes without generating or rewriting map assets."""


def validate_fixed_route_slice(available, count, start):
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (available, count, start)):
        raise ValueError('Fixed-route counts/offset must be integers')
    if available <= 0 or count <= 0 or start < 0 or start+count > available:
        raise ValueError(f'Requested routes [{start},{start+count}) exceed {available} saved routes')


def fixed_route_indices(env_ids, start):
    """Same source row for start, goal, direction and initial orientation."""
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError('Fixed-route offset must be a nonnegative integer')
    return env_ids.detach().cpu().long()+start
