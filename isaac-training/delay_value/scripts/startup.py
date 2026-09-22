"""Share Isaac startup helpers with training, including older local revisions."""
from shared import load_training_module


_startup = load_training_module("startup")
navigation_asset_imports = _startup.navigation_asset_imports


def navigation_viewport_enabled(cfg):
    if hasattr(_startup, "navigation_viewport_enabled"):
        return _startup.navigation_viewport_enabled(cfg)
    return bool(cfg.get("viewer", {}).get(
        "enable_viewport", not cfg.headless or cfg.get("eval", {}).get("record_video", False)))


def clear_navigation_simulation(env):
    if hasattr(_startup, "clear_navigation_simulation"):
        return _startup.clear_navigation_simulation(env)
    annotator = getattr(env, "_rgb_annotator", None)
    if annotator is not None:
        annotator.detach()
        env._rgb_annotator = None
    product = getattr(env, "_render_product", None)
    if product is not None and hasattr(product, "destroy"):
        product.destroy()
        env._render_product = None
    env.sim.stop()
    env.sim.clear_all_callbacks()
    env.sim.clear()
    env.sim.clear_instance()
    from omni_drones.robots.robot import RobotBase
    RobotBase._robots.clear()
