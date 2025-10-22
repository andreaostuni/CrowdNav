try:
    # Gymnasium renamed the package while keeping a compatible API.
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover - fallback for legacy gym installs
    from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
    disable_env_checker=True,
    order_enforce=False,
)
