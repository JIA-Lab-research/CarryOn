"""Command-line entry point for CarryOn."""
import argparse
from . import __version__


def _validate_simulation_policy(config, registry, *, allow_simulation):
    ros = config.default_transport == "ros" or config.ros_force_enable or config.callback_ros_enabled or config.result_channel == "ros" or any(ep.transport == "ros" for ep in registry.list())
    if ros:
        raise ValueError("ROS is not included in the simulation release")
    remote = [ep.name for ep in registry.list() if ep.transport not in ("mock", "local")]
    if (remote or config.recovery.enabled) and not allow_simulation:
        raise ValueError("this profile enables simulator HTTP endpoints/recovery; pass --allow-simulation")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="carryon", description="CarryOn agent")
    parser.add_argument("--version", action="version", version=f"CarryOn {__version__}")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run the configured agent")
    run.add_argument("--config", required=True)
    run.add_argument("--allow-simulation", action="store_true", help="Enable simulator HTTP endpoints")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from .config import Config
    from .endpoints import load_endpoints_yaml
    config = Config.from_file(args.config)
    registry = load_endpoints_yaml(config.endpoints_config)
    try:
        _validate_simulation_policy(config, registry, allow_simulation=args.allow_simulation)
    except ValueError as error:
        parser.error(str(error))
    from .main import main as agent_main
    return agent_main(["--config", args.config] + (["--allow-simulation"] if args.allow_simulation else []))


if __name__ == "__main__":
    raise SystemExit(main())
